import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from analysis import fig7_models as m
from analysis import fig7_learning_efficiency as f
from analysis import fig7_extended_efficiency as ext
from src.cl_tsmnet import training as t


@pytest.mark.parametrize('kind',['bilstm','tahag','lsccn'])
def test_additional_checkpoint_reconstruction_and_native_prediction(tmp_path,kind):
    torch.set_num_threads(1);torch.manual_seed(42)
    if kind=='bilstm': original=t.build_temporal_baseline(kind,14,128,2)
    elif kind=='tahag': original=t.build_tahag(14,2)
    else: original=t.build_lsccn(14,14+len(t.BFGCN_FEATURE_BANDS),2)
    original.eval();fold=tmp_path/'subject_01';fold.mkdir()
    cp=fold/'model.pt';torch.save(original.state_dict(),cp)
    pd.DataFrame([dict(subject=1,decision_threshold=.125)]).to_csv(tmp_path/'summary.csv',index=False)
    x=np.random.default_rng(42).normal(size=(2,14,128)).astype(np.float32)
    ds=dict(x=x,y=np.array([0,1]));device=torch.device('cpu')
    restored,_=m.build(dict(model_type=kind),ds,dict(source_ids=np.arange(2)),cp,device)
    inputs=m.prepare_input(kind,x,[1,1],128.,device)
    with torch.no_grad():
        native=m.forward(original,kind,inputs)
        actual=f.audited_fp32_forward(m.convert_fp32(restored,kind,device),kind,inputs,device)
    torch.testing.assert_close(actual,native)
    if kind=='lsccn':
        scores=np.array([[.2,.3],[.2,.325],[.3,.2]])
        assert m.predictions(restored,kind,scores).tolist()==[0,1,0]
        pd.DataFrame([dict(subject=1)]).to_csv(tmp_path/'summary.csv',index=False)
        with pytest.raises(ValueError,match='decision_threshold'):
            m.build(dict(model_type=kind),ds,dict(source_ids=np.arange(2)),cp,device)


def test_plot_fixture_only(tmp_path):
    rows=[]
    for ds in ['stew','eegmat']:
        for i,method in enumerate(ext.ORDER):
            if ds=='eegmat' and method=='LSCCN':continue
            rows.append(dict(dataset=ds,method=method,mean_target_bacc=.6+i*.02,
                             std_target_bacc=.1,latency_median_ms=.1+i*.25))
    ext.plot(pd.DataFrame(rows),tmp_path)
    assert (tmp_path/'Fig7_extended_efficiency.pdf').stat().st_size>1000


def test_bool_config_is_not_python_string_truthiness():
    assert m.value({'flag':'False'},'flag',True) is False


def test_preserved_measurements_are_read_only_and_corruption_is_rejected(tmp_path):
    runs=[];result_files=[]
    for ds in ['stew','eegmat']:
        subjects=list(range(1,49)) if ds=='stew' else list(range(36))
        for kind,method in ext.ORIGINAL.items():
            runs.append(dict(dataset=ds,model_type=kind,subjects=subjects))
            for subject in subjects:
                folder=tmp_path/'benchmarks'/ds/kind/f'subject_{subject:02d}';folder.mkdir(parents=True)
                p=folder/'latency_ms.npy';np.save(p,np.array([.1,.2]))
                result=dict(signature={'repeats':2},latency_sha256=f.sha(p),row=dict(
                    dataset=ds,method=method,fold_id=subject,fp32_prediction_disagreement=0))
                f.write_json(folder/'result.json',result);result_files.append(folder/'result.json')
    f.write_json(tmp_path/'fig7_runs.json',runs)
    before=[f.sha(p) for p in result_files]
    _,rows,_=ext.read_original(tmp_path)
    assert len(rows)==504 and before==[f.sha(p) for p in result_files]
    p=tmp_path/'benchmarks/stew/eegnet/subject_01/latency_ms.npy';np.save(p,np.array([9.,10.]))
    with pytest.raises(ValueError,match='integrity'):ext.read_original(tmp_path)
