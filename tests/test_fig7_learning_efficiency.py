from argparse import Namespace
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
import torch
from analysis import fig7_learning_efficiency as f
from analysis import fig7_models as m
from src.cl_tsmnet import training as t


def test_coverage_no_carry_forward_and_e95(tmp_path):
    folder=tmp_path/'run'; subjects=list(range(5))
    for s in subjects:
        p=folder/('subject_%02d'%s); p.mkdir(parents=True)
        vals=[.5,.8,.9] if s<3 else [.5,.8]
        pd.DataFrame(dict(epoch=np.arange(1,len(vals)+1),val_bacc=vals,train_loss=1.)).to_csv(p/'history.csv',index=False)
    args=Namespace(output_dir=str(tmp_path),seed=42,bootstrap_replicates=100,trust_legacy_source_validation=True)
    runs=[dict(model_type='eegconformer',dataset='stew',missing=False,folder=str(folder),subjects=subjects)]
    f.convergence(runs,args)
    result=pd.read_csv(tmp_path/'fig7_convergence_epochwise.csv')
    assert result.n_folds.tolist()==[5,5,3]
    assert result.eligible.tolist()==[True,True,False]
    assert result.mean_val_bacc.tolist()==pytest.approx([.5,.8,.9])
    assert pd.read_csv(tmp_path/'fig7_convergence_per_fold.csv').E95.tolist()==[3,3,3,2,2]
    args.trust_legacy_source_validation=False
    with pytest.raises(ValueError,match='provenance'): f.convergence(runs,args)


@pytest.mark.parametrize('kind',['eegnet','eegconformer','bfgcn','mdtn','tsmnet','ms_tgc_spddsbn'])
def test_real_model_fp32_port_cpu(kind):
    torch.set_num_threads(1); torch.manual_seed(42)
    device=torch.device('cpu'); c,s=14,128; domains=np.array([1,2,3])
    if kind=='eegnet': model=t.build_eegnet(c,s,2)
    elif kind=='eegconformer': model=t.build_eegconformer(c,s,2,depth=1)
    elif kind=='bfgcn': model=t.build_bfgcn(c,2)
    elif kind=='mdtn': model=t.build_mdtn_gmda(c,2)
    elif kind=='tsmnet': model=t.build_tsmnet(str(f.ROOT),c,s,2,domains,bnorm='spddsbn',device=device)
    else: model=t.build_ms_tgc_spddsbn(str(f.ROOT),c,s,2,domains,device=device,variant=kind)
    model.eval()
    windows=np.random.default_rng(42).normal(size=(1,c,s)).astype(np.float32)
    inputs=m.prepare_input(kind,windows,[3],128.,device)
    with torch.no_grad(): before=m.forward(model,kind,inputs).detach()
    model=m.convert_fp32(model,kind,device)
    with torch.no_grad(): after=f.audited_fp32_forward(model,kind,inputs,device)
    assert after.dtype==torch.float32 and torch.isfinite(after).all()
    torch.testing.assert_close(after,before.float(),atol=2e-3,rtol=2e-3)


def test_new_training_epoch_log_and_source_audit(tmp_path):
    torch.set_num_threads(1)
    ds=dict(name='stew',fs=128.,x=np.random.default_rng(1).normal(size=(12,14,128)).astype(np.float32),
            y=np.tile([0,1],6),meta=pd.DataFrame(dict(subject=np.repeat([1,2,3],4))))
    result=t.train_one_split(ds,np.repeat([1,2,3],4),dict(train=np.arange(4),val=np.arange(4,8),test=np.arange(8,12)),
                             str(f.ROOT),output_dir=str(tmp_path),model_type='eegnet',epochs=1,batch_size=2)
    log=pd.read_csv(tmp_path/'epoch_metrics.csv'); audit=json.loads((tmp_path/'source_validation_audit.json').read_text())
    assert log.source_val_bacc.iloc[0]==pytest.approx(result['history'][0]['val_bacc'])
    assert log.elapsed_training_seconds.iloc[0]>0
    assert audit['split_subjects']['val']==[2] and audit['split_subjects']['test']==[3]


def test_repair_commands_parse_and_preserve_recorded_protocol(tmp_path,monkeypatch):
    import run_experiment
    import sys
    record=dict(dataset='stew',model_type='eegconformer',epochs=17,batch_size=32,
                lr=.002,augment=False,target_adapt=False,seed=9,output_dir='outputs/stew_loso_eegconformer')
    runs=[dict(dataset='stew',model_type='eegconformer',missing=True,record=record),
          dict(dataset='stew',model_type='mdtn',missing=True)]
    args=Namespace(output_dir=str(tmp_path),repair_root='outputs/fig7_test_repair',data_root='data',cache_root='outputs/cache')
    f.prepare_repairs(runs,args)
    plan=json.loads((tmp_path/'fig7_repair_plan.json').read_text())
    for entry in plan:
        monkeypatch.setattr(sys,'argv',['run_experiment.py']+entry['command'][3:])
        parsed=run_experiment.parse_args()
        assert parsed.protocol=='loso' and parsed.epochs==17 and parsed.batch_size==32 and parsed.seed==9
        assert parsed.no_augment and parsed.subject is None


def test_source_audit_rejects_target_overlap(tmp_path):
    fold=tmp_path/'subject_01'; fold.mkdir()
    pd.DataFrame(dict(epoch=[1],train_loss=[1.],val_bacc=[.5])).to_csv(fold/'history.csv',index=False)
    (fold/'source_validation_audit.json').write_text(json.dumps(dict(split_subjects=dict(train=[2],val=[1],test=[1]),
       target_labels_for_selection=False,selection_metric='source_val_loss')))
    run=dict(dataset='stew',model_type='eegconformer',missing=False,folder=str(tmp_path),subjects=[1])
    args=Namespace(output_dir=str(tmp_path),seed=42,bootstrap_replicates=100,trust_legacy_source_validation=False)
    with pytest.raises(ValueError,match='audit failed'): f.convergence([run],args)


def test_plot_layout_fixture_only_not_scientific_results(tmp_path):
    # Explicit layout fixture stored only in pytest temp; never a publication result.
    curves=[]; efficiency=[]
    for dataset in ['stew','eegmat']:
        for i,(kind,method) in enumerate(f.METHODS.items()):
            efficiency.append(dict(dataset=dataset,method=method,precision='fp32',hardware='TEST FIXTURE',
                                   params_M=.05*(i+1),latency_median_ms=.5+i*.1,mean_target_bacc=.7+i*.01))
            if kind in f.CURVES:
                for epoch in [1,2,3]:
                    curves.append(dict(dataset=dataset,method=method,epoch=epoch,eligible=True,
                        mean_val_bacc=.6+.03*epoch,ci_low=.55+.03*epoch,ci_high=.65+.03*epoch))
    pd.DataFrame(curves).to_csv(tmp_path/'fig7_convergence_epochwise.csv',index=False)
    pd.DataFrame(efficiency).to_csv(tmp_path/'fig7_efficiency_summary.csv',index=False)
    f.plot(Namespace(output_dir=str(tmp_path)))
    assert (tmp_path/'Fig7_learning_computational_efficiency.pdf').stat().st_size>1000
    assert (tmp_path/'fig7_plot_audit.json').exists()
