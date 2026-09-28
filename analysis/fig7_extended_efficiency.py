"""Extend completed six-model measurements without rerunning or overwriting them."""
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from analysis import fig7_learning_efficiency as f

ORIGINAL=dict(f.METHODS)
EXTRA={'bilstm':'BiLSTM','tahag':'TAHAG','lsccn':'LSCCN'}
ALL={**ORIGINAL,**EXTRA}
ORDER=['EEGNet','BiLSTM','TAHAG','LSCCN','EEG-Conformer','BF-GCN','MDTN-GMDA','TSMNet','AGMNet']


def read_original(root):
    runs=json.loads((root/'fig7_runs.json').read_text(encoding='utf-8'))
    rows=[]; provenance=[]
    for run in runs:
        if run.get('missing') or run['model_type'] not in ORIGINAL:
            raise ValueError('Original six-model run is incomplete')
        for subject in run['subjects']:
            folder=root/'benchmarks'/run['dataset']/run['model_type']/f'subject_{subject:02d}'
            result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
            path=folder/'latency_ms.npy'
            times=np.load(path,allow_pickle=False)
            if f.sha(path)!=result['latency_sha256'] or len(times)!=result['signature']['repeats'] or not np.isfinite(times).all() or np.any(times<=0):
                raise ValueError('Original timing integrity failure: '+str(folder))
            row=result['row']
            if (row['dataset'],row['method'],row['fold_id'])!=(run['dataset'],ORIGINAL[run['model_type']],subject):
                raise ValueError('Original measurement identity mismatch')
            if row['fp32_prediction_disagreement']!=0:
                raise ValueError('Original FP32 predictions differ')
            row=dict(row,timing_file=str(path.resolve()),measurement_origin='preserved_original')
            rows.append(row)
            provenance.append(dict(result_file=str((folder/'result.json').resolve()),
                result_sha256=f.sha(folder/'result.json'),signature=result['signature']))
    expected={(ds,m) for ds in ['stew','eegmat'] for m in ORIGINAL.values()}
    if {(r['dataset'],r['method']) for r in rows}!=expected:
        raise ValueError('Original results must contain both datasets and all six methods')
    for ds,m in expected:
        ids=[r['fold_id'] for r in rows if r['dataset']==ds and r['method']==m]
        if sorted(ids)!=(list(range(1,49)) if ds=='stew' else list(range(36))):
            raise ValueError('Original fold set is incomplete or duplicated')
    return runs,rows,provenance


def discover_extra(args):
    runs=[];issues=[]
    old_methods=f.METHODS
    try:
        for ds in ['stew','eegmat']:
            f.METHODS={k:v for k,v in EXTRA.items() if ds=='stew' or k!='lsccn'}
            f.MODEL_NAMES.update({k:{k} for k in EXTRA})
            local=copy.copy(args);local.datasets=ds
            found,problems=f.discover(local);runs.extend(found);issues.extend(problems)
    finally:
        f.METHODS=old_methods
    return runs,issues


def aggregate(rows,hardware):
    records=[]
    for (ds,method),g in pd.DataFrame(rows).groupby(['dataset','method']):
        times=np.concatenate([np.load(p,allow_pickle=False) for p in g.timing_file])
        records.append(dict(dataset=ds,method=method,n_folds=len(g),mean_target_bacc=g.target_bacc.mean(),
            std_target_bacc=g.target_bacc.std(ddof=1),params_M=g.params_M.mean(),
            latency_median_ms=float(np.median(times)),latency_mean_ms=float(times.mean()),
            latency_std_ms=float(times.std(ddof=1)),latency_IQR_ms=float(np.subtract(*np.percentile(times,[75,25]))),
            precision='fp32',hardware=hardware))
    return pd.DataFrame(records)


def plot(frame,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox
    colors=dict(zip(ORDER,['#777777','#332288','#44AA99','#882255','#4477AA','#AA3377','#228833','#AA9900','#CC6677']))
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42}):
        fig,axes=plt.subplots(2,2,figsize=(8.2,6.5),layout='constrained')
        label_groups=[]
        for col,ds in enumerate(['stew','eegmat']):
            g=frame[frame.dataset==ds].set_index('method').reindex(ORDER)
            ax=axes[0,col]
            for j,method in enumerate(ORDER):
                row=g.loc[method]
                if pd.isna(row.mean_target_bacc):
                    ax.text(51,j,'not available',color='#777777',fontsize=7)
                    continue
                ax.errorbar(row.mean_target_bacc*100,j,xerr=row.std_target_bacc*100,
                            fmt='o',color=colors[method],capsize=2,ms=5)
            ax.set_yticks(range(len(ORDER)),ORDER);ax.invert_yaxis();ax.set_xlim(35,100)
            ax.set_xlabel('Target BAcc (%) — mean ± subject SD')
            ax.set_title(f'({"ab"[col]}) {ds.upper()}');ax.grid(axis='x',alpha=.18)
            ax=axes[1,col]
            annotations=[];points=[]
            for method,row in g.dropna(subset=['mean_target_bacc']).iterrows():
                x=row.latency_median_ms;y=row.mean_target_bacc*100
                ax.scatter(x,y,s=40,color=colors[method],edgecolor='black',linewidth=.4)
                annotations.append(ax.annotate(method,(x,y),xytext=(5,6),
                            textcoords='offset points',fontsize=6.5,
                            arrowprops=dict(arrowstyle='-',color=colors[method],lw=.4)))
                points.append((x,y))
            ax.set_xscale('log');ax.margins(x=.45,y=.25)
            ax.set_xlabel('Inference latency (ms/window; log scale)');ax.set_ylabel('Target BAcc (%)')
            ax.set_title(f'({"cd"[col]}) {ds.upper()}');ax.grid(alpha=.18)
            label_groups.append((ax,annotations,points))
        fig.suptitle('Predictive performance and computational cost',fontsize=11)
        fig.supxlabel('FP32, batch=1. BF-GCN / TAHAG / LSCCN feature preprocessing excluded.\n'
                      'SVM not measured; EEGMAT LSCCN unavailable. Independent methods are not connected by curves.',fontsize=7)
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        for ax,annotations,points in label_groups:
            occupied=[Bbox.from_bounds(px-5,py-5,10,10) for px,py in ax.transData.transform(points)]
            for ann in annotations:
                candidates=[]
                for dy in [6,-12,18,-24,30,-36,42,-48]:
                    for dx,ha in [(7,'left'),(-7,'right'),(22,'left'),(-22,'right')]:
                        ann.set_position((dx,dy));ann.set_ha(ha);ann.update_positions(renderer)
                        box=matplotlib.text.Text.get_window_extent(ann,renderer).expanded(1.06,1.12)
                        collisions=sum(box.overlaps(other) for other in occupied)
                        inside=ax.bbox.contains(box.x0,box.y0) and ax.bbox.contains(box.x1,box.y1)
                        candidates.append((collisions+10*(not inside),abs(dx)+abs(dy),dx,dy,ha,box))
                best=min(candidates,key=lambda c:c[:2])
                ann.set_position(best[2:4]);ann.set_ha(best[4]);occupied.append(best[5])
                if best[0]: print('Layout warning: inspect label',ann.get_text())
        for ext in ['pdf','png']:
            fig.savefig(out/f'Fig7_extended_efficiency.{ext}',dpi=600)
        plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['audit','all','plot'],default='audit')
    p.add_argument('--existing-results',default='results/fig7_fixed_v2')
    p.add_argument('--master-summary',default='outputs/master_summary.csv,outputs/fig7_retrained/master_summary.csv')
    p.add_argument('--run-config',help='Explicit paths for additional models only; multiple candidates otherwise fail')
    p.add_argument('--output-root',default='outputs');p.add_argument('--output-dir',default='results/fig7_extended')
    p.add_argument('--cache-root',default='outputs/cache_fig6_rebuilt');p.add_argument('--data-root',default='data')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--threads',type=int,default=1)
    p.add_argument('--warmup',type=int,default=100);p.add_argument('--repeats',type=int,default=1000)
    p.add_argument('--eval-batch-size',type=int,default=16);p.add_argument('--seed',type=int,default=42)
    args=p.parse_args();out=Path(args.output_dir);old=Path(args.existing_results)
    if out.resolve()==old.resolve() or old.resolve() in out.resolve().parents:
        p.error('Use an independent output directory outside the preserved results')
    if args.warmup<100 or args.repeats<1000 or min(args.threads,args.eval_batch_size)<1:p.error('Invalid measurement budget')
    out.mkdir(parents=True,exist_ok=True)
    if args.stage=='plot':
        plot(pd.read_csv(out/'fig7_efficiency_summary.csv'),out);return
    original,rows,provenance=read_original(old)
    runs,issues=discover_extra(args)
    paths=[]
    for run in original+runs:
        paths.append(dict(dataset=run['dataset'],method=ALL[run['model_type']],
                          folder=run.get('folder','MISSING'),action='preserve' if run in original else 'benchmark'))
    pd.DataFrame(paths).to_csv(out/'fig7_selected_paths.csv',index=False)
    print(pd.DataFrame(paths).to_string(index=False),flush=True)
    f.write_json(out/'fig7_additional_runs.json',runs);f.write_json(out/'fig7_extension_issues.json',issues)
    f.write_json(out/'fig7_preserved_provenance.json',provenance)
    for issue in issues:print(issue['severity'].upper(),issue['dataset'],issue['method'],issue['message'])
    if any(i['severity']=='error' for i in issues):raise ValueError('Resolve input errors before benchmarking')
    # Confirm saved measurements refer to the checkpoint files still selected on this server.
    for run in original:
        for subject in run['subjects']:
            cp=Path(run['folder'])/f'subject_{subject:02d}'/'model.pt'
            row=next(r for r in rows if (r['dataset'],r['method'],r['fold_id'])==(run['dataset'],ORIGINAL[run['model_type']],subject))
            if f.sha(cp)!=row['checkpoint_sha256']:raise ValueError('Preserved checkpoint differs: '+str(cp))
    if args.stage=='audit':return
    device=torch.device('cuda:0' if args.device=='cuda' else 'cpu');torch.set_num_threads(args.threads)
    env=f.hardware(device)
    for item in provenance:
        sig=item['signature']
        if sig['environment']!=env or sig['warmup']!=args.warmup or sig['repeats']!=args.repeats or sig['eval_batch_size']!=args.eval_batch_size:
            raise ValueError('Hardware/software or timing protocol differs from preserved measurements; no silent pooling')
    for ds in ['stew','eegmat']:
        local=argparse.Namespace(protocol='loso',data_root=args.data_root,cache_root=args.cache_root,
                                 target_fs_stew=None,target_fs_eegmat=None,target_fs_cog_bci=None)
        context=f.f5._load_dataset_context(f.f5._parse_datasets(ds,ds.upper())[0],local)
        cache_sha=f.sha(context['cache'])
        if any(i['signature']['dataset_cache']!=cache_sha for i in provenance if i['signature']['record']['dataset']==ds):
            raise ValueError('Dataset cache differs from original: '+ds)
    local=copy.copy(args);local.output_dir=str(out/'additional_measurements')
    Path(local.output_dir).mkdir(parents=True,exist_ok=True)
    f.METHODS.update(EXTRA)
    f.benchmark(runs,local)
    new=pd.read_csv(Path(local.output_dir)/'fig7_efficiency_per_fold.csv')
    for row in new.to_dict('records'):
        kind=next(k for k,v in EXTRA.items() if v==row['method'])
        row['timing_file']=str((Path(local.output_dir)/'benchmarks'/row['dataset']/kind/f"subject_{int(row['fold_id']):02d}"/'latency_ms.npy').resolve())
        row['measurement_origin']='additional_measurement';rows.append(row)
    pd.DataFrame(rows).to_csv(out/'fig7_efficiency_per_fold.csv',index=False)
    result=aggregate(rows,env['gpu'] or env['cpu']);result.to_csv(out/'fig7_efficiency_summary.csv',index=False)
    plot(result,out)


if __name__=='__main__':main()
