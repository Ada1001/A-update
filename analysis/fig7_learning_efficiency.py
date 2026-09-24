"""Real LOSO learning curves and checkpoint inference benchmarks for Fig.7."""
import argparse
import hashlib
import json
import math
import platform
import shlex
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from analysis import fig5_representation_alignment as f5
from analysis import fig7_models as models

METHODS={'eegnet':'EEGNet','eegconformer':'EEG-Conformer','bfgcn':'BF-GCN',
         'mdtn':'MDTN-GMDA','tsmnet':'TSMNet','ms_tgc_spddsbn':'AGMNet'}
CURVES=['eegconformer','mdtn','tsmnet','ms_tgc_spddsbn']
COLORS=['#777777','#4477AA','#AA3377','#228833','#CCBB44','#CC6677']
MODEL_NAMES={k:{k} for k in METHODS}
MODEL_NAMES['mdtn']={'mdtn','mdtn_gmda'}
MODEL_NAMES['tsmnet']={'tsmnet_spddsbn'}


def canonical_type(value):
    return 'mdtn' if value in {'mdtn','mdtn_gmda'} else value


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=str,allow_nan=False),encoding='utf-8')


def clean(record):
    return {k:v for k,v in record.items() if not pd.isna(v)}


def resolve_directory(record,args):
    original=Path(str(record['output_dir']).replace('\\','/'))
    # Preserve an existing exact path (including repaired nested runs).
    relocated=Path(args.output_root)/original.name
    if original.is_dir(): return original
    if relocated.is_dir(): return relocated
    return relocated


def discover(args):
    frames=[]
    for p in args.master_summary.split(','):
        frame=pd.read_csv(p).copy(); frame['_master']=str(Path(p).resolve()); frames.append(frame)
    master=pd.concat(frames,ignore_index=True)
    overrides=json.loads(Path(args.run_config).read_text(encoding='utf-8')) if args.run_config else {}
    runs=[]; issues=[]
    for dataset in args.datasets.split(','):
        for kind,name in METHODS.items():
            subset=master[(master.dataset==dataset)&(master.protocol=='loso')&(master.model_type.map(canonical_type)==kind)]
            subset=subset[subset.model.isin(MODEL_NAMES[kind])] # exact aliases only; excludes ablations
            choices=overrides.get(dataset,{})
            override=choices.get(kind,choices.get('mdtn_gmda') if kind=='mdtn' else None)
            candidates=subset.output_dir.astype(str).unique().tolist()
            if override:
                subset=subset[subset.output_dir.astype(str).str.replace('\\','/',regex=False).str.rstrip('/')==str(override).replace('\\','/').rstrip('/')]
            if len(subset)==0:
                message=('Configured output_dir not found: '+str(override)+'; available matching runs: '+str(candidates)) if override and candidates else 'No canonical LOSO master record (accepted model names: '+','.join(sorted(MODEL_NAMES[kind]))+')'
                issues.append(dict(dataset=dataset,method=name,severity='error',message=message))
                runs.append(dict(dataset=dataset,model_type=kind,missing=True)); continue
            if len(subset)>1:
                distinct=subset.output_dir.astype(str).nunique()
                if distinct>1 and not override:
                    issues.append(dict(dataset=dataset,method=name,severity='error',message='Multiple run directories; choose explicitly in --run-config'))
                    runs.append(dict(dataset=dataset,model_type=kind,missing=True)); continue
                issues.append(dict(dataset=dataset,method=name,severity='warning',message='Repeated master rows for same directory; latest timestamp selected and checked against fold summary'))
                subset=subset.sort_values('timestamp',kind='stable')
            record=clean(subset.iloc[-1].to_dict()); folder=resolve_directory(record,args)
            record['_original_model_type']=record['model_type']
            record['model_type']=kind
            run=dict(dataset=dataset,model_type=kind,record=record,folder=str(folder),missing=False)
            summary_path=folder/'summary.csv'
            if not summary_path.exists():
                issues.append(dict(dataset=dataset,method=name,severity='error',message='Missing '+str(summary_path)))
                run['missing']=True; runs.append(run); continue
            frame=pd.read_csv(summary_path)
            if not {'subject','test_bacc'}<=set(frame) or frame.subject.duplicated().any():
                raise ValueError('Invalid fold summary: '+str(summary_path))
            for col,expected in [('dataset',dataset),('model_type',kind),('protocol','loso')]:
                values=frame[col].map(canonical_type) if col=='model_type' and col in frame else frame.get(col,pd.Series(dtype=str))
                if col not in frame or set(values)!={expected}: raise ValueError('Fold summary identity mismatch: '+str(summary_path))
            subjects=sorted(frame.subject.astype(int).tolist()); run['subjects']=subjects
            run['summary_sha256']=sha(summary_path)
            if len(subjects)!=int(record['n']) or not np.isclose(frame.test_bacc.mean(),float(record['balanced_accuracy_mean']),atol=1e-7):
                issues.append(dict(dataset=dataset,method=name,severity='error',message='Master summary and fold summary disagree; cannot pair this configuration with checkpoints'))
            legacy_subjects=[]
            for subject in subjects:
                fold=folder/('subject_%02d'%subject)
                for filename in ['model.pt']+(['history.csv'] if kind in CURVES else []):
                    if not (fold/filename).exists() and not (filename=='history.csv' and (fold/'epoch_metrics.csv').exists()):
                        issues.append(dict(dataset=dataset,method=name,severity='error',message='Missing '+str(fold/filename)))
                if kind in CURVES and not (fold/'source_validation_audit.json').exists():
                    legacy_subjects.append(subject)
            if legacy_subjects:
                issues.append(dict(dataset=dataset,method=name,severity='warning',subjects=legacy_subjects,
                    message=f'{len(legacy_subjects)} folds lack legacy source-validation provenance; requires --trust-legacy-source-validation'))
            runs.append(run)
        good=[r for r in runs if r['dataset']==dataset and not r['missing']]
        if good:
            ref=good[0]
            for r in good[1:]:
                if r['subjects']!=ref['subjects']:
                    issues.append(dict(dataset=dataset,method=METHODS[r['model_type']],severity='error',message='LOSO subject set differs between methods'))
                keys=['seed','epochs','patience','batch_size','lr','weight_decay','val_size','augment','target_fs','artifact_z']
                differences=[key for key in keys if key in r['record'] and key in ref['record'] and r['record'][key]!=ref['record'][key]]
                unknown=[key for key in keys if (key in r['record'])!=(key in ref['record'])]
                if differences:
                    issues.append(dict(dataset=dataset,method=METHODS[r['model_type']],severity='warning',message='Training protocol differs from '+METHODS[ref['model_type']]+': '+','.join(differences)))
                if unknown:
                    issues.append(dict(dataset=dataset,method=METHODS[r['model_type']],severity='warning',message='Training protocol cannot be compared because a record omits: '+','.join(unknown)))
    return runs,issues


def convergence(runs,args):
    epochs=[]; fold_stats=[]; raw=[]; warnings=[]
    rng=np.random.default_rng(args.seed)
    for run in runs:
        kind=run['model_type']
        if kind not in CURVES or run['missing']: continue
        histories=[]
        for subject in run['subjects']:
            folder=Path(run['folder'])/('subject_%02d'%subject)
            p=folder/'epoch_metrics.csv'
            if not p.exists(): p=folder/'history.csv'
            h=pd.read_csv(p)
            metric='source_val_bacc' if 'source_val_bacc' in h else 'val_bacc'
            if not {'epoch',metric,'train_loss'}<=set(h) or h.empty: raise ValueError('Incomplete history: '+str(p))
            if h.epoch.duplicated().any() or h.epoch.tolist()!=list(range(1,len(h)+1)):
                raise ValueError('Epochs must be contiguous real records from 1: '+str(p))
            if not np.isfinite(h[metric]).all() or not h[metric].between(0,1).all(): raise ValueError('BAcc must be a finite fraction')
            if 'source_val_bacc' in h and 'val_bacc' in h and not np.allclose(h.source_val_bacc,h.val_bacc):
                raise ValueError('Conflicting source validation metrics: '+str(p))
            fold_summary=Path(run['folder'])/'summary.csv'
            if fold_summary.exists():
                completed=pd.read_csv(fold_summary)
                if 'epochs_ran' in completed and len(h)!=int(completed.loc[completed.subject==subject,'epochs_ran'].iloc[0]):
                    raise ValueError('History is truncated or belongs to another run: '+str(p))
            if 'elapsed_training_seconds' in h:
                elapsed=h.elapsed_training_seconds.to_numpy()
                if not np.isfinite(elapsed).all() or np.any(elapsed<0) or np.any(np.diff(elapsed)<0):
                    raise ValueError('Invalid measured training time: '+str(p))
            audit_file=folder/'source_validation_audit.json'
            if audit_file.exists():
                audit=json.loads(audit_file.read_text(encoding='utf-8')); ss=audit['split_subjects']
                if (set(ss['val'])&set(ss['test']) or set(ss['train'])&set(ss['test']) or
                    audit['target_labels_for_selection'] or audit['selection_metric']!='source_val_loss'):
                    raise ValueError('Source-validation audit failed: '+str(folder))
                audit_status='recorded_source_validation'
            else:
                if not args.trust_legacy_source_validation:
                    raise ValueError('Legacy validation provenance unknown: '+str(p)+'; review original code before --trust-legacy-source-validation')
                audit_status='user_asserted_legacy_source_validation_NOT_independently_verified'
            h=h.copy(); h['source_val_bacc']=h[metric]
            h['dataset']=run['dataset']; h['method']=METHODS[kind]; h['fold_id']=subject
            h['validation_audit']=audit_status
            if 'elapsed_training_seconds' not in h:
                h['elapsed_training_seconds']=np.nan
                warnings.append(f"{run['dataset']} {kind} S{subject}: historical elapsed time unavailable; not inferred")
            raw.append(h); histories.append(h)
            threshold=.95*h.source_val_bacc.max()
            fold_stats.append(dict(dataset=run['dataset'],method=METHODS[kind],fold_id=subject,
                                   E95=int(h.loc[h.source_val_bacc>=threshold,'epoch'].iloc[0]),stopping_epoch=int(h.epoch.max())))
        all_h=pd.concat(histories)
        for epoch,group in all_h.groupby('epoch'):
            vals=group.source_val_bacc.to_numpy(); n=len(vals)
            draws=vals[rng.integers(0,n,size=(args.bootstrap_replicates,n))].mean(1)
            lo,hi=np.quantile(draws,[.025,.975])
            epochs.append(dict(dataset=run['dataset'],method=METHODS[kind],epoch=int(epoch),mean_val_bacc=float(vals.mean()),
                               ci_low=lo,ci_high=hi,n_folds=n,total_folds=len(histories),
                               eligible=n>=math.ceil(.8*len(histories))))
    out=Path(args.output_dir)
    pd.DataFrame(epochs).to_csv(out/'fig7_convergence_epochwise.csv',index=False)
    if raw: pd.concat(raw).to_csv(out/'fig7_epoch_records.csv',index=False)
    frame=pd.DataFrame(fold_stats); summaries=[]
    if len(frame):
        for (dataset,method),g in frame.groupby(['dataset','method']):
            q1,q3=np.quantile(g.E95,[.25,.75])
            summaries.append(dict(dataset=dataset,method=method,median_E95=g.E95.median(),IQR_E95=q3-q1,E95_q25=q1,E95_q75=q3,
                                  median_stopping_epoch=g.stopping_epoch.median(),n_folds=len(g)))
        frame.to_csv(out/'fig7_convergence_per_fold.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'fig7_convergence_summary.csv',index=False)
    return warnings


def hardware(device):
    props=torch.cuda.get_device_properties(device) if device.type=='cuda' else None
    return dict(device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                host=platform.node(),gpu_uuid=str(getattr(props,'uuid','unavailable')),
                gpu_memory_bytes=getattr(props,'total_memory',None),
                torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                python=platform.python_version(),platform=platform.platform(),cpu=platform.processor(),
                threads=torch.get_num_threads(),precision='FP32_all_floating_parameters_and_forward',
                batch_size=1,tf32=False,autocast=False,timer='perf_counter_ns with device synchronization per forward')


def audited_fp32_forward(model,kind,inputs,device):
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves
    class FP32Audit(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            output=func(*args,**(kwargs or {}))
            for v in tree_leaves((args,kwargs,output)):
                if isinstance(v,torch.Tensor) and v.is_floating_point() and (v.dtype!=torch.float32 or v.device!=device):
                    raise ValueError('Mixed dtype/device operation: '+str(func)+' '+str(v.dtype)+' '+str(v.device))
            return output
    with FP32Audit(): return models.forward(model,kind,inputs)


def benchmark(runs,args):
    device=torch.device('cuda:0' if args.device=='cuda' else 'cpu')
    if device.type=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable; no silent CPU fallback')
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    env=hardware(device); out=Path(args.output_dir); write_json(out/'fig7_environment.json',env)
    if device.type!='cuda' or '4090 D' not in str(env['gpu']) and '4090D' not in str(env['gpu']):
        print('WARNING: measured hardware is not the requested RTX 4090D:',env['gpu'] or env['cpu'],flush=True)
    fold_rows=[]; contexts={}
    for run in runs:
        if run['missing']: continue
        dsname,kind=run['dataset'],run['model_type']; record=run['record']
        if dsname not in contexts:
            local=argparse.Namespace(protocol='loso',cache_root=args.cache_root,data_root=args.data_root,
                                     target_fs_stew=None,target_fs_eegmat=None,target_fs_cog_bci=None)
            spec=f5._parse_datasets(dsname,dsname.upper())[0]
            contexts[dsname]=f5._load_dataset_context(spec,local)
            contexts[dsname]['cache_sha256']=sha(contexts[dsname]['cache'])
        context=contexts[dsname]; ds=context['dataset_object']
        if value_float(record,'target_fs',ds['fs'])!=float(ds['fs']): raise ValueError('Dataset sampling rate mismatch')
        if sorted(np.unique(ds['meta'].subject).astype(int).tolist())!=run['subjects']:
            raise ValueError('Dataset subject universe differs from run: '+run['folder'])
        summary=pd.read_csv(Path(run['folder'])/'summary.csv')
        for subject in run['subjects']:
            print('Benchmark',dsname,kind,subject,flush=True)
            checkpoint=Path(run['folder'])/('subject_%02d'%subject)/'model.pt'
            cp_sha=sha(checkpoint); split=f5._make_split_context(context,subject,f5._split_config(dict(record=record,summary=summary)))
            if set(ds['meta'].iloc[split['val_ids']].subject)&set(ds['meta'].iloc[split['target_ids']].subject):
                raise ValueError('Source validation and target subjects overlap')
            cache=out/'benchmarks'/dsname/kind/('subject_%02d'%subject); cache.mkdir(parents=True,exist_ok=True)
            signature=dict(checkpoint=cp_sha,dataset_cache=context['cache_sha256'],record=record,environment=env,
                           warmup=args.warmup,repeats=args.repeats,adapter_sha=sha(ROOT/'analysis/fig7_models.py'),
                           runner_sha=sha(__file__),eval_batch_size=args.eval_batch_size)
            result_file=cache/'result.json'; latency_file=cache/'latency_ms.npy'
            if result_file.exists():
                saved=json.loads(result_file.read_text())
                if saved['signature']==signature and saved['latency_sha256']==sha(latency_file):
                    fold_rows.append(saved['row']); print('  reused audited timing',flush=True); continue
                raise ValueError('Timing provenance changed; choose a fresh output directory: '+str(cache))
            model,migrations=models.build(record,ds,split,checkpoint,device)
            ids=split['target_ids']; windows=split['normalizer'].transform_array(ds['x'][ids]).astype(np.float32)
            domains=split['domains'][ids]
            # Preprocessing and device transfers are entirely outside latency measurement.
            native=[]
            with torch.no_grad():
                for start in range(0,len(ids),args.eval_batch_size):
                    inputs=models.prepare_input(kind,windows[start:start+args.eval_batch_size],domains[start:start+args.eval_batch_size],ds['fs'],device)
                    logits=models.forward(model,kind,inputs)
                    if not torch.isfinite(logits).all(): raise ValueError('Non-finite native checkpoint output')
                    native.append(logits.detach().cpu().numpy())
            native=np.concatenate(native); native_pred=native.argmax(1)
            native_bacc=balanced_accuracy_score(ds['y'][ids],native_pred)
            reported=float(summary.loc[summary.subject==subject,'test_bacc'].iloc[0])
            if abs(native_bacc-reported)>1e-6: raise ValueError(f'Checkpoint BAcc {native_bacc} != stored {reported}; verify data/config/code, not a timing failure')
            model=models.convert_fp32(model,kind,device)
            converted=[]
            with torch.no_grad():
                for start in range(0,len(ids),args.eval_batch_size):
                    inputs=models.prepare_input(kind,windows[start:start+args.eval_batch_size],domains[start:start+args.eval_batch_size],ds['fs'],device)
                    converted.append(models.forward(model,kind,inputs).detach().cpu().numpy())
            converted=np.concatenate(converted)
            if not np.isfinite(converted).all(): raise ValueError('FP32 port unstable; cannot publish uniform-FP32 timing for this checkpoint')
            disagreement=float(np.mean(converted.argmax(1)!=native_pred))
            if disagreement>0: raise ValueError(f'FP32 changes {disagreement:.2%} target predictions; refusing to pair native accuracy with FP32 latency')
            inputs=models.prepare_input(kind,windows[:1],domains[:1],ds['fs'],device)
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            # Observe every dispatched floating tensor once; hooks/timing audits excluded from timing.
            def sync():
                if device.type=='cuda': torch.cuda.synchronize(device)
            with torch.no_grad():
                checked=audited_fp32_forward(model,kind,inputs,device)
                if not np.array_equal(checked.argmax(1).cpu().numpy(),native_pred[:1]): raise ValueError('Batch-one prediction mismatch')
                for _ in range(args.warmup): models.forward(model,kind,inputs)
                sync(); latency=[]
                for _ in range(args.repeats):
                    sync(); start=time.perf_counter_ns(); logits=models.forward(model,kind,inputs); sync()
                    latency.append((time.perf_counter_ns()-start)/1e6)
            if any(not torch.equal(v.detach().cpu(),state[k]) for k,v in model.state_dict().items()): raise ValueError('Inference changed model state')
            if sha(checkpoint)!=cp_sha: raise ValueError('Checkpoint changed during benchmarking')
            params=sum(p.numel() for p in model.parameters() if p.requires_grad)
            row=dict(dataset=dsname,method=METHODS[kind],fold_id=subject,target_bacc=native_bacc,
                     params_M=params/1e6,trainable_parameters=params,latency_median_ms=float(np.median(latency)),
                     latency_mean_ms=float(np.mean(latency)),latency_std_ms=float(np.std(latency,ddof=1)),
                     latency_IQR_ms=float(np.subtract(*np.percentile(latency,[75,25]))),
                     fp32_prediction_disagreement=disagreement,fp32_max_logit_difference=float(np.max(np.abs(converted-native))),
                     raw_window_shape=list(windows[:1].shape),model_input_shapes=[list(x.shape) for x in inputs],
                     checkpoint_sha256=cp_sha,precision='fp32',migrations=migrations)
            np.save(latency_file,np.asarray(latency)); write_json(result_file,dict(signature=signature,row=row,latency_sha256=sha(latency_file)))
            fold_rows.append(row)
            del model
    pd.DataFrame(fold_rows).to_csv(out/'fig7_efficiency_per_fold.csv',index=False)
    aggregate=[]
    for (dataset,method),g in pd.DataFrame(fold_rows).groupby(['dataset','method']):
        kind=next(k for k,v in METHODS.items() if v==method)
        times=np.concatenate([np.load(out/'benchmarks'/dataset/kind/('subject_%02d'%s)/'latency_ms.npy') for s in g.fold_id])
        aggregate.append(dict(dataset=dataset,method=method,n_folds=len(g),mean_target_bacc=g.target_bacc.mean(),
            std_target_bacc=g.target_bacc.std(ddof=1),params_M=g.params_M.mean(),params_M_min=g.params_M.min(),params_M_max=g.params_M.max(),
            latency_median_ms=np.median(times),latency_mean_ms=times.mean(),latency_std_ms=times.std(ddof=1),
            latency_IQR_ms=np.subtract(*np.percentile(times,[75,25])),precision='fp32',hardware=env['gpu'] or env['cpu']))
    pd.DataFrame(aggregate).to_csv(out/'fig7_efficiency_summary.csv',index=False)


def value_float(record,key,default): return float(record.get(key,default))


def prepare_repairs(runs,args):
    """Write reviewable commands; never start training during analysis/audit."""
    import run_experiment
    # Avoid assuming the results directory depth; require launch from project root.
    commands=['#!/usr/bin/env bash','set -euo pipefail',
              'test -f run_experiment.py || { echo "Run this script from the project root"; exit 1; }',
              'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1']
    repair_root=Path(args.repair_root); mappings={}; planned=[]
    for r in runs:
        kind=r['model_type']; ds=r['dataset']
        missing=r['missing'] or any(not (Path(r['folder'])/('subject_%02d'%s)/'model.pt').exists() or
            (kind in CURVES and not any((Path(r['folder'])/('subject_%02d'%s)/h).exists() for h in ['history.csv','epoch_metrics.csv']))
            for s in r.get('subjects',[]))
        if not missing: continue
        own=r.get('record')
        reference=own or next((x.get('record') for x in runs if x['dataset']==ds and x['model_type']=='eegconformer' and x.get('record')),None)
        if reference is None: continue
        old_argv=sys.argv
        try:
            sys.argv=['run_experiment.py','--dataset',ds,'--protocol','loso']
            defaults=vars(run_experiment.parse_args())
        finally: sys.argv=old_argv
        values=dict(defaults)
        # Original model record preferred; missing model inherits only training protocol.
        shared=['epochs','patience','batch_size','refit_batch_size','lr','weight_decay','seed','val_size','single_val_size','test_size','target_fs','artifact_z']
        keys=defaults.keys() if own else shared
        for key in keys:
            if key in reference and key not in ['dataset','model','model_name','output','master_summary','subject','cache','cache_root','data_root','protocol']:
                v=reference[key]
                if defaults.get(key) is not None and not isinstance(defaults[key],bool): v=type(defaults[key])(v)
                values[key]=v
        values.update(dataset=ds,model=kind,protocol='loso',bnorm='spddsbn',output=str(repair_root),
                      master_summary=str(repair_root/'master_summary.csv'),cache_root=args.cache_root,data_root=args.data_root)
        values['no_augment']=not truth(reference.get('augment',True))
        values['no_target_adapt']=not truth(reference.get('target_adapt_requested',reference.get('target_adapt',True))) if own else False
        # model-specific architecture/config not copied from an unrelated model.
        cmd=['/root/miniconda3/bin/python','-u','run_experiment.py']
        ignored={'subject','cache','model_name','rebuild_cache','allow_incomplete_splits'}
        for key,v in values.items():
            if key in ignored or v is None: continue
            flag='--'+key.replace('_','-')
            if isinstance(v,bool):
                if v: cmd.append(flag)
            else: cmd.extend([flag,str(v)])
        from src.cl_tsmnet.experiment_utils import run_directory_name
        run_name=run_directory_name(ds,'loso',run_experiment._model_label(argparse.Namespace(**values)),values['bnorm'])
        destination=repair_root/run_name
        commands += [f'test ! -e {shlex.quote(str(destination))} || {{ echo "Refusing to overwrite {destination}"; exit 1; }}',
                     'mkdir -p '+shlex.quote(str(repair_root)),shlex.join(cmd)+' 2>&1 | tee '+shlex.quote(str(repair_root/(ds+'_'+kind+'.log')))]
        mappings.setdefault(ds,{})[kind]=str(destination)
        planned.append(dict(dataset=ds,method=METHODS[kind],configuration_source='own master record' if own else 'same-dataset EEG-Conformer protocol; method defaults',
                            note='Recreates recorded fields; historical unrecorded implementation/configuration cannot be guaranteed identical',command=cmd))
    out=Path(args.output_dir)
    (out/'fig7_retrain_missing.sh').write_text('\n'.join(commands)+'\n',encoding='utf-8')
    write_json(out/'fig7_repair_plan.json',planned)
    # Include explicit choices for existing methods to avoid ambiguity after merging masters.
    for r in runs:
        if r.get('record') and r['model_type'] not in mappings.get(r['dataset'],{}):
            mappings.setdefault(r['dataset'],{})[r['model_type']]=r['record']['output_dir']
    write_json(out/'fig7_repaired_run_config.json',mappings)


def truth(value):
    return str(value).lower() in ['true','1','1.0']


def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    out=Path(args.output_dir); curves=pd.read_csv(out/'fig7_convergence_epochwise.csv'); eff=pd.read_csv(out/'fig7_efficiency_summary.csv')
    if set(eff.precision)!={'fp32'} or eff.hardware.nunique(dropna=False)!=1:
        raise ValueError('Mixed hardware/precision in efficiency CSV')
    if not np.isfinite(eff[['params_M','latency_median_ms','mean_target_bacc']]).all().all() or (eff.latency_median_ms<=0).any():
        raise ValueError('Invalid efficiency numbers')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(7.16,5.1)); markers=['o','s','^','D','v','P']; styles=['-','--','-.',':','--','-']
    label_groups=[]
    for col,dataset in enumerate(['stew','eegmat']):
        ax=axes[0,col]
        for kind in CURVES:
            name=METHODS[kind]; idx=list(METHODS).index(kind)
            g=curves[(curves.dataset==dataset)&(curves.method==name)&curves.eligible.astype(bool)].sort_values('epoch')
            if g.empty: raise ValueError('Missing eligible convergence data: '+dataset+' '+name)
            ax.plot(g.epoch,g.mean_val_bacc*100,color=COLORS[idx],ls=styles[idx],lw=1.7 if kind=='ms_tgc_spddsbn' else 1.,
                    marker=markers[idx],ms=2.8,markevery=max(1,len(g)//5))
            ax.fill_between(g.epoch.to_numpy(),g.ci_low.to_numpy()*100,g.ci_high.to_numpy()*100,color=COLORS[idx],alpha=.12,lw=0)
        ax.set(title='('+chr(97+col)+') '+dataset.upper(),xlabel='Epoch',ylabel='Source-validation BAcc (%)')
        ax=axes[1,col]; g=eff[eff.dataset==dataset]
        if set(g.method)!=set(METHODS.values()): raise ValueError('Trade-off requires all six methods: '+dataset)
        if g.latency_median_ms.max()/g.latency_median_ms.min()>10: ax.set_xscale('log')
        annotations=[]
        for idx,(kind,name) in enumerate(METHODS.items()):
            r=g[g.method==name].iloc[0]; area=float(np.clip(25+45*np.sqrt(r.params_M),25,140))
            ax.scatter(r.latency_median_ms,r.mean_target_bacc*100,s=area,marker=markers[idx],color=COLORS[idx],
                       edgecolors='black',linewidths=1.1 if kind=='ms_tgc_spddsbn' else .35)
            annotations.append(ax.annotate(name,(r.latency_median_ms,r.mean_target_bacc*100),xytext=(5,6 if idx%2==0 else -12),
                        textcoords='offset points',fontsize=6))
        label_groups.append((ax,annotations))
        ax.margins(x=.3,y=.25)
        ax.set(title='('+chr(99+col)+') '+dataset.upper(),xlabel='Inference latency (ms/window)',ylabel='Cross-subject BAcc (%)')
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.18,lw=.4)
    handles=[Line2D([],[],color=COLORS[i],marker=markers[i],ls=styles[i],label=name,ms=4) for i,name in enumerate(METHODS.values())]
    fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False)
    fig.text(.5,.015,'Area = clip(25 + 45 sqrt(parameters in M), 25, 140). FP32 forward only; BF-GCN preprocessing excluded.',ha='center',fontsize=5.5)
    fig.tight_layout(rect=(0,.045,1,.89),h_pad=1.5)
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    for ax,annotations in label_groups:
        occupied=[]
        for ann in annotations:
            chosen=None; best=None
            for dy in [6,-12,18,-24,30,-36]:
                for dx,ha in [(5,'left'),(-5,'right')]:
                    ann.set_position((dx,dy)); ann.set_ha(ha)
                    box=ann.get_window_extent(renderer).expanded(1.06,1.15)
                    score=sum(box.overlaps(other) for other in occupied)+10*int(not ax.bbox.contains(box.x0,box.y0) or not ax.bbox.contains(box.x1,box.y1))
                    if best is None or score<best: best=score; chosen=(dx,dy,ha)
            dx,dy,ha=chosen; ann.set_position((dx,dy)); ann.set_ha(ha)
            occupied.append(ann.get_window_extent(renderer).expanded(1.06,1.15))
            if best: print('WARNING: inspect trade-off label overlap:',ann.get_text(),flush=True)
    for ext in ['pdf','png']: fig.savefig(out/('Fig7_learning_computational_efficiency.'+ext),dpi=600)
    plt.close(fig)
    write_json(out/'fig7_plot_audit.json',dict(source_csv_sha256={p.name:sha(p) for p in [out/'fig7_convergence_epochwise.csv',out/'fig7_efficiency_summary.csv']},
                                             scaling='CSV BAcc fractions multiplied by 100 only',curve_filter='eligible >=80%; no carry-forward'))


def report(args,runs,issues):
    text=['# Learning and Computational Efficiency — audit and interpretation',
          'Only real histories/checkpoints are accepted. Missing data block the complete figure. No synthetic results or carried-forward epochs.',
          'Validation selection in current training.py minimizes source-validation loss, not target accuracy. Historical logs without audit sidecars require an explicit user assertion; this cannot establish the provenance of past code.',
          'Bootstrap ribbons resample real folds at each epoch (>=80% coverage); overlapping LOSO training sets make these descriptive conditional intervals, not independent-replication evidence. E95 is the first observed epoch >=95% of that fold\'s best observed validation BAcc.',
          'Old histories may lack elapsed time; missing time is NaN, never estimated from epoch or inference speed.',
          'Uniform-FP32 benchmarking is an explicit inference port of the saved native-precision checkpoint. Native TSMNet/AGMNet use CPU FP64 SPD operations. The benchmark copy moves all floating state and SPD inference to the selected device/FP32; checkpoint files and training code precision are unchanged. All target predictions must match native inference and native BAcc must match the saved summary. Failures stop the benchmark rather than hiding precision changes.',
          'BF-GCN consumes precomputed bandpower/PLV derived from the same real 1-second window. These CPU feature computations are excluded as preprocessing. Its forward-only latency is not end-to-end raw-EEG latency. All methods include their implemented forward branches; no BF-GCN domain branch is manually removed.',
          'Every fold: eval + no_grad, batch 1, >=100 warmups, >=1000 repeats, perf_counter_ns with CUDA synchronization around each call; inputs are already on device. Timing pools equal repeat counts across folds. Domain adaptation/refitting is outside timing. Parameters count only requires_grad=True; fold min/max recorded.',
          '## Actual AGMNet computation',
          'Let T be raw temporal samples, C channels, F features, M scales, K Chebyshev terms, |E| graph edges, d SPD subspace; T\' is pooled graph time (default64). Shared temporal convolutions cost O(C T F sum_m kernel_m). Scale attention operates on M pooled tokens per channel: approximately O(C M F^2 + C M^2 F), not attention on all T samples.',
          'ChebyGraphSequenceLayer._propagate uses dense torch.einsum("nm,bmft->bnft",...). Top-k adjacency masking does NOT make computation sparse. Propagation is O(K C^2 F T\'); learned feature projections add O(K C F^2 T\') when widths match. Do not claim O(K |E| F) sparse execution for this implementation.',
          'GraphSPDManifoldHead forms moments over C*T\' observations, O(F^2 C T\'), and AugSPD has size (F+1)x(F+1). With F=64 this is65, reduced by BiMap to d (default20), cost O((F+1)^2 d+(F+1)d^2). ReEig, LogEig and SPD normalization spectral operations scale cubically in d per decomposition. Frozen-stat eval does not run the iterative Karcher refit loop; refitting has extra data/iteration-dependent cost. Values must be checked against each selected run, not assumed from names.',
          '## Recorded warnings / blockers']
    text += [f"- {i['severity'].upper()} {i['dataset']} {i['method']}: {i['message']}" for i in issues]
    eff=Path(args.output_dir)/'fig7_efficiency_summary.csv'
    if eff.exists():
        text+=['## Measured results',pd.read_csv(eff).to_string(index=False),
               'Report accuracy, latency and parameters together. These measurements do not by themselves establish fastest, real-time, significantly better, or a favorable trade-off. No subjective Pareto line is drawn.']
    (Path(args.output_dir)/'FIG7_EFFICIENCY_ANALYSIS.md').write_text('\n\n'.join(text),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['audit','convergence','benchmark','plot','all'],default='audit')
    p.add_argument('--master-summary',default='outputs/master_summary.csv',help='Comma-separated master CSVs')
    p.add_argument('--output-root',default='outputs'); p.add_argument('--output-dir',default='results/fig7')
    p.add_argument('--data-root',default='data'); p.add_argument('--cache-root',default='outputs/cache')
    p.add_argument('--datasets',default='stew,eegmat'); p.add_argument('--run-config',help='JSON dataset -> model_type -> original output_dir in master')
    p.add_argument('--repair-root',default='outputs/fig7_retrained')
    p.add_argument('--trust-legacy-source-validation',action='store_true')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda'); p.add_argument('--threads',type=int,default=1)
    p.add_argument('--warmup',type=int,default=100); p.add_argument('--repeats',type=int,default=1000)
    p.add_argument('--eval-batch-size',type=int,default=16); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--bootstrap-replicates',type=int,default=5000)
    args=p.parse_args()
    if args.warmup<100 or args.repeats<1000: p.error('Protocol requires warmup>=100 and repeats>=1000')
    if min(args.threads,args.eval_batch_size)<1 or args.bootstrap_replicates<100: p.error('Invalid thread/batch/bootstrap count')
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    runs,issues=discover(args); write_json(out/'fig7_runs.json',runs); write_json(out/'fig7_input_audit.json',issues)
    prepare_repairs(runs,args)
    report(args,runs,issues)
    for i in issues: print(i['severity'].upper(),i['dataset'],i['method'],i['message'])
    if args.stage=='audit': return
    if any(i['severity']=='error' for i in issues): raise ValueError('Input audit has blockers; see fig7_input_audit.json')
    try:
        if args.stage in ['convergence','all']:
            for warning in convergence(runs,args):
                issues.append(dict(dataset='logs',method='all',severity='warning',message=warning))
        if args.stage in ['benchmark','all']: benchmark(runs,args)
        if args.stage in ['plot','all']: plot(args)
    except Exception as exc:
        issues.append(dict(dataset='pipeline',method='all',severity='error',message=str(exc)))
        raise
    finally:
        write_json(out/'fig7_input_audit.json',issues); report(args,runs,issues)


if __name__=='__main__': main()
