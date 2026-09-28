from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import pytest
from analysis import fig7_learning_reporting as reporting
from analysis import fig7_extended_efficiency as ext
from analysis import fig7_learning_efficiency as f


def test_late_selected_lsccn_threshold_is_not_applied_to_history():
    run=dict(dataset='stew',model_type='lsccn',record={})
    h=pd.DataFrame(dict(epoch=[1,2],val_loss=[.8,.6],val_bacc=[.5,.6]))
    s=pd.DataFrame([dict(subject=1,best_epoch=2,best_val_loss=.6,decision_threshold=.12)])
    result=reporting.audit_history(run,Path('subject_01'),h,s)
    assert 'threshold 0' in result['interpretation']
    assert result['final_decision_threshold']==.12 and result['val_stat_refit']=='unknown'
    s.loc[0,'best_epoch']=1
    with pytest.raises(ValueError,match='Best epoch'):reporting.audit_history(run,Path('subject_01'),h,s)


def test_reporting_all_histories_and_preserves_inputs(tmp_path,monkeypatch):
    # Synthetic records only for pipeline tests, never publication data.
    original=tmp_path/'original';extended=tmp_path/'extended';out=tmp_path/'report'
    original.mkdir();extended.mkdir();old=[];new=[];records=[];summary_rows=[];sources=[]
    for ds in ['stew','eegmat']:
        subjects=list(range(1,49)) if ds=='stew' else list(range(36))
        for kind,method in ext.ALL.items():
            if ds=='eegmat' and kind=='lsccn':continue
            folder=tmp_path/'runs'/f'{ds}_{kind}';folder.mkdir(parents=True)
            run=dict(dataset=ds,model_type=kind,subjects=subjects,record={},folder=str(folder),missing=False)
            (new if kind in ext.EXTRA else old).append(run)
            rows=[]
            for subject in subjects:
                fold=folder/f'subject_{subject:02d}';fold.mkdir()
                (fold/'model.pt').write_bytes(b'test fixture checkpoint')
                history=pd.DataFrame(dict(epoch=[1,2],train_loss=[.8,.6],val_loss=[.8,.6],val_bacc=[.5,.6]))
                history.to_csv(fold/'history.csv',index=False)
                sources.append(fold/'history.csv')
                rows.append(dict(subject=subject,epochs_ran=2,best_epoch=2,best_val_loss=.6,test_bacc=.7))
                records.append(dict(dataset=ds,method=method,fold_id=subject,target_bacc=.7,checkpoint_sha256=f.sha(fold/'model.pt')))
            pd.DataFrame(rows).to_csv(folder/'summary.csv',index=False)
            summary_rows.append(dict(dataset=ds,method=method,n_folds=len(subjects),mean_target_bacc=.7,std_target_bacc=0.,latency_median_ms=1.,precision='fp32'))
    f.write_json(original/'fig7_runs.json',old);f.write_json(extended/'fig7_additional_runs.json',new)
    pd.DataFrame(records).to_csv(extended/'fig7_efficiency_per_fold.csv',index=False)
    pd.DataFrame(summary_rows).to_csv(extended/'fig7_efficiency_summary.csv',index=False)
    before={str(p):f.sha(p) for p in sources}
    monkeypatch.setattr(sys,'argv',['report','--stage','audit','--original-results',str(original),
        '--extended-results',str(extended),'--output-dir',str(out),'--bootstrap-replicates','100','--trust-legacy-source-validation'])
    reporting.main()
    result=pd.read_csv(out/'fig7_convergence_epochwise.csv')
    assert len(result.groupby(['dataset','method']))==13
    assert result[result.epoch==2].mean_val_bacc.to_numpy()==pytest.approx(np.full(13,.6))
    assert before=={str(p):f.sha(p) for p in sources}
    broken=pd.DataFrame(summary_rows);broken.loc[0,'mean_target_bacc']=.9
    with pytest.raises(ValueError,match='differs from folds'):
        reporting.validate_efficiency(broken,pd.DataFrame(records))


def test_main_and_supplement_layout_fixture_only(tmp_path):
    efficiency=[];curves=[]
    for ds in ['stew','eegmat']:
        for i,name in enumerate(ext.ORDER):
            if ds=='eegmat' and name=='LSCCN':continue
            efficiency.append(dict(dataset=ds,method=name,mean_target_bacc=.65+.01*i,std_target_bacc=.1,latency_median_ms=.1+.25*i))
            if name not in ['EEGNet','BF-GCN']:
                for epoch in range(1,6):
                    curves.append(dict(dataset=ds,method=name,epoch=epoch,mean_val_bacc=.5+.02*epoch+.01*i,
                                       ci_low=.48+.02*epoch+.01*i,ci_high=.52+.02*epoch+.01*i,eligible=epoch<5))
    ext.plot(pd.DataFrame(efficiency),tmp_path,curves=pd.DataFrame(curves),basename='main_fixture')
    ext.plot(pd.DataFrame(efficiency),tmp_path,basename='supplement_fixture')
    assert (tmp_path/'main_fixture.pdf').stat().st_size>1000
    assert (tmp_path/'supplement_fixture.pdf').stat().st_size>1000
