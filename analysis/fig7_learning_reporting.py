"""Produce learning-curve main Fig7 and final-performance supplement without timing or training."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from analysis import fig7_learning_efficiency as f
from analysis import fig7_extended_efficiency as ext

CURVE_KINDS=['bilstm','tahag','lsccn','eegconformer','mdtn','tsmnet','ms_tgc_spddsbn']


def validate_efficiency(efficiency,per_fold):
    expected={(ds,name) for ds in ['stew','eegmat'] for name in ext.ORDER if not (ds=='eegmat' and name=='LSCCN')}
    if set(zip(efficiency.dataset,efficiency.method))!=expected or efficiency.duplicated(['dataset','method']).any():
        raise ValueError('Expected nine STEW and eight EEGMAT methods in efficiency summary')
    if set(zip(per_fold.dataset,per_fold.method))!=expected:
        raise ValueError('Per-fold efficiency methods differ from summary')
    if set(efficiency.precision)!={'fp32'}:raise ValueError('Efficiency precision is not uniformly FP32')
    if not np.isfinite(efficiency[['mean_target_bacc','std_target_bacc','latency_median_ms']]).all().all() or (efficiency.latency_median_ms<=0).any():
        raise ValueError('Invalid efficiency metrics')
    for row in efficiency.itertuples():
        folds=per_fold[(per_fold.dataset==row.dataset)&(per_fold.method==row.method)]
        ids=list(range(1,49)) if row.dataset=='stew' else list(range(36))
        if sorted(folds.fold_id.tolist())!=ids or row.n_folds!=len(ids):
            raise ValueError('Efficiency fold set mismatch: '+row.dataset+' '+row.method)
        if not np.isfinite(folds.target_bacc).all() or not folds.target_bacc.between(0,1).all():
            raise ValueError('Invalid per-fold BAcc')
        if not np.isclose(row.mean_target_bacc,folds.target_bacc.mean(),atol=1e-10,rtol=1e-10) or not np.isclose(row.std_target_bacc,folds.target_bacc.std(ddof=1),atol=1e-10,rtol=1e-10):
            raise ValueError('Efficiency summary differs from folds: '+row.dataset+' '+row.method)


def audit_history(run,fold,frame,summary):
    required={'epoch','val_loss'}
    if not required<=set(frame) or frame.empty:
        raise ValueError('History missing epochs/validation loss: '+str(fold))
    if not np.isfinite(frame.val_loss).all():raise ValueError('Non-finite validation loss: '+str(fold))
    if 'validation_scope' in frame and set(frame.validation_scope.dropna())-{'source_validation'}:
        raise ValueError('History explicitly records a non-source validation scope: '+str(fold))
    if 'selection_metric' in frame and set(frame.selection_metric.dropna())-{'source_val_loss'}:
        raise ValueError('History selection metric differs from source validation loss: '+str(fold))
    selected=summary.loc[summary.subject==int(fold.name.removeprefix('subject_'))]
    if len(selected)!=1:raise ValueError('Missing or duplicate fold summary: '+str(fold))
    row=selected.iloc[0]
    best=int(frame.loc[frame.val_loss.idxmin(),'epoch'])
    if 'best_epoch' in row and pd.notna(row.best_epoch) and int(row.best_epoch)!=best:
        raise ValueError('Best epoch differs from history: '+str(fold))
    if 'best_val_loss' in row and pd.notna(row.best_val_loss) and not np.isclose(float(row.best_val_loss),frame.val_loss.min(),rtol=1e-6,atol=1e-7):
        raise ValueError('Best validation loss differs from history: '+str(fold))
    kind=run['model_type']
    refit=row.get('val_stat_refit',run['record'].get('val_stat_refit','unknown'))
    if pd.isna(refit):refit='unknown'
    if kind=='lsccn':
        interpretation='Current code: epoch validation uses score difference threshold 0; final evaluation uses validation-selected threshold. Historical code needs provenance.'
    elif kind in {'tsmnet','ms_tgc_spddsbn'}:
        interpretation='Domain-statistic refit affects validation; missing historical refit flag is unknown, not false.'
    elif kind=='tahag':
        interpretation='Current code: eval(), source bandpower features, no target batch in validation forward; training may use unlabeled target adaptation.'
    elif kind=='bilstm':
        interpretation='Current code: eval(), raw normalized windows, source-validation classification; historical code needs provenance.'
    else:
        interpretation='Recorded source-validation learning metric; distinct from final adapted target performance.'
    return dict(dataset=run['dataset'],method=ext.ALL[kind],fold_id=int(row.subject),
                best_epoch=best,epochs_ran=len(frame),val_stat_refit=refit,
                final_decision_threshold=row.get('decision_threshold',None),
                provenance_sidecar=(fold/'source_validation_audit.json').is_file(),
                interpretation=interpretation)


def collect_runs(original_root,extended_root):
    original=json.loads((original_root/'fig7_runs.json').read_text(encoding='utf-8'))
    additional=json.loads((extended_root/'fig7_additional_runs.json').read_text(encoding='utf-8'))
    runs=[r for r in original+additional if r['model_type'] in CURVE_KINDS]
    expected={(ds,k) for ds in ['stew','eegmat'] for k in CURVE_KINDS if not (ds=='eegmat' and k=='lsccn')}
    keys=[(r['dataset'],r['model_type']) for r in runs]
    if set(keys)!=expected or len(keys)!=len(set(keys)) or any(r.get('missing') for r in runs):
        raise ValueError('Learning curves require 13 unique dataset/method runs, including all five added runs')
    return runs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-results',default='results/fig7_fixed_v2')
    p.add_argument('--extended-results',default='results/fig7_extended_v2')
    p.add_argument('--output-dir',default='results/fig7_two_versions')
    p.add_argument('--stage',choices=['audit','all'],default='all')
    p.add_argument('--trust-legacy-source-validation',action='store_true')
    p.add_argument('--bootstrap-replicates',type=int,default=5000)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.bootstrap_replicates<100:p.error('bootstrap-replicates must be at least 100')
    original=Path(args.original_results);extended=Path(args.extended_results);out=Path(args.output_dir)
    for source in [original,extended]:
        if out.resolve()==source.resolve() or source.resolve() in out.resolve().parents:
            p.error('Use a separate output directory outside source results')
    out.mkdir(parents=True,exist_ok=True)
    runs=collect_runs(original,extended)
    per_fold=pd.read_csv(extended/'fig7_efficiency_per_fold.csv')
    efficiency=pd.read_csv(extended/'fig7_efficiency_summary.csv')
    validate_efficiency(efficiency,per_fold)
    audits=[];sources=[]
    for run in runs:
        folder=Path(run['folder']);summary=pd.read_csv(folder/'summary.csv')
        expected=list(range(1,49)) if run['dataset']=='stew' else list(range(36))
        if sorted(run['subjects'])!=expected or sorted(summary.subject.tolist())!=expected:
            raise ValueError('Incomplete or duplicate subject set: '+str(folder))
        method=ext.ALL[run['model_type']]
        for subject in run['subjects']:
            fold=folder/f'subject_{subject:02d}'
            measurement=per_fold[(per_fold.dataset==run['dataset'])&(per_fold.method==method)&(per_fold.fold_id==subject)]
            if len(measurement)!=1 or f.sha(fold/'model.pt')!=measurement.checkpoint_sha256.iloc[0]:
                raise ValueError('Learning history directory does not match timed checkpoint: '+str(fold))
            path=fold/'epoch_metrics.csv'
            if not path.exists():path=fold/'history.csv'
            h=pd.read_csv(path)
            if 'epoch_metrics.csv'==path.name and (fold/'history.csv').exists():
                other=pd.read_csv(fold/'history.csv')
                for col in ['epoch','val_bacc','val_loss']:
                    if col in h and col in other and (len(h)!=len(other) or not np.allclose(h[col],other[col],rtol=1e-6,atol=1e-7)):
                        raise ValueError('Conflicting history/epoch_metrics: '+str(fold))
            audits.append(audit_history(run,fold,h,summary))
            sources.append(dict(path=str(path),sha256=f.sha(path)))
    pd.DataFrame(audits).to_csv(out/'fig7_validation_protocol_audit.csv',index=False)
    f.write_json(out/'fig7_learning_sources.json',sources)
    old_methods,old_curves=f.METHODS,f.CURVES
    try:
        f.METHODS=ext.ALL;f.CURVES=CURVE_KINDS
        warnings=f.convergence(runs,args)
    finally:
        f.METHODS=old_methods;f.CURVES=old_curves
    curves=pd.read_csv(out/'fig7_convergence_epochwise.csv')
    notes=['# Fig7 learning / efficiency audit',
        'Main: real source-validation BAcc by epoch, plus existing target-accuracy/median-latency scatter. Supplement: final target BAcc mean +/- subject SD, plus the same scatter.',
        'No training, timing, smoothing, target-label tuning or early-stopped-epoch extrapolation is performed. At least 80% fold coverage is required at each plotted epoch.',
        'Bootstrap intervals describe recorded folds. LOSO training sets overlap; these are not independent population-replication intervals. E95 is relative to each fold own validation peak, not evidence of comparable learning efficiency.',
        'LSCCN: current training evaluates epoch BAcc with threshold 0; final threshold is selected afterward using validation labels. Do not compare the historical epoch metric with final threshold-adjusted accuracy as if they used the same decision rule.',
        'TSMNet/AGMNet: validation-statistics refit flags must be considered. Missing legacy flags are unknown; current code cannot prove historical protocol. Low source-validation accuracy does not prove low adapted-target accuracy.',
        'Old six and added models were measured on different recorded GPU UUIDs. Retain the provenance; matching model names alone do not establish identical physical measurement conditions.',
        'Accepted TAHAG BAcc discrepancies remain unresolved: tolerance acceptance does not establish rounding as the cause. The supplementary and main efficiency panels reuse the recomputed values.',
        'Historical elapsed times are missing for some methods. Epoch is not wall-clock training time. No formal cross-method convergence-speed claim is established by these figures.',
        '## Per-method source checks',pd.DataFrame(audits).groupby(['dataset','method','provenance_sidecar']).size().to_string(),
        '## Recorded warnings']+warnings
    (out/'FIG7_TWO_VERSIONS_REVIEW.md').write_text('\n\n'.join(notes),encoding='utf-8')
    print(pd.DataFrame(audits).groupby(['dataset','method']).size().to_string())
    if args.stage=='all':
        ext.plot(efficiency,out,basename='FigS7_final_performance')
        ext.plot(efficiency,out,curves=curves,basename='Fig7_learning_and_efficiency')
    f.write_json(out/'fig7_reporting_audit.json',dict(stage=args.stage,seed=args.seed,
        curve_methods=CURVE_KINDS,trust_legacy_source_validation=args.trust_legacy_source_validation,
        source_summary_sha256=f.sha(extended/'fig7_efficiency_summary.csv'),
        source_per_fold_sha256=f.sha(extended/'fig7_efficiency_per_fold.csv'),
        validation_audit_sha256=f.sha(out/'fig7_validation_protocol_audit.csv')))


if __name__=='__main__':main()
