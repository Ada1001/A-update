"""Synthetic correctness/layout fixtures only; never publication observations."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from analysis import fig6_node_response_topomaps as f


def test_norm_contrast_and_consistency():
    values=np.array([[[3.,4.],[0.,2.]],[[0.,5.],[0.,8.]]])
    response=f.compute_node_response(values)
    np.testing.assert_allclose(response,[[5,2],[5,8]])
    result=f.compute_subject_contrast_topomap(response,np.array([0,1]),1)
    np.testing.assert_allclose(result["delta"],[0,6])
    maps=np.array([[1,2,3],[2,4,6],[3,2,1],[1,1,1]])
    stats=f.compute_topomap_consistency(maps)
    assert stats["valid_pairs"]==3
    assert stats["mean"]==pytest.approx(-1/3)
    assert stats["constant_subjects"]==[3]
    with pytest.raises(ValueError):
        f.compute_subject_contrast_topomap(response,np.array([0,0]),1)


def test_unique_selection_with_ties_and_explicit():
    scores=pd.DataFrame(dict(subject_id=[5,1,7,2,9],bacc=[.8]*5))
    chosen=f.select_representative_subjects(scores)
    assert [x["subject_id"] for x in chosen]==[1,2,5,7]
    assert [x["subject_id"] for x in f.select_representative_subjects(scores,[9,7,2,1])]==[9,7,2,1]
    with pytest.raises(ValueError): f.select_representative_subjects(scores,[1,1,2,5])


def test_hooks_same_forward_no_state_changes():
    method=f.f5._load_methods(None)[-1]
    ds=dict(x=np.zeros((12,4,32),np.float32),y=np.tile([0,1],6),fs=128.)
    config=f.f5._model_config(dict(mstgc_temporal_hidden=4,mstgc_graph_hidden=4,mstgc_fusion_dim=7,
        mstgc_num_heads=2,mstgc_time_points=8,subspacedims=3,mstgc_dropout=0.))
    domains=np.repeat([1,2,3],4)
    model=f.f5._build_model(ds,domains,np.arange(12),np.arange(4),method,config,torch.device("cpu"))
    x=torch.randn(4,4,32); d=torch.ones(4,dtype=torch.long)
    before={k:v.clone() for k,v in model.state_dict().items()}
    with torch.no_grad(): baseline=model(x,d)
    pre,post,logits=f.extract_pre_post_node_features(model,x,d)
    assert pre.shape==post.shape==(4,4,32)
    torch.testing.assert_close(logits,baseline,rtol=0,atol=0)
    for k,v in before.items(): torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
    assert not model.graph._forward_hooks and not model.graph._forward_pre_hooks


def test_checkpoint_extraction_and_cache_identity(tmp_path):
    ds=dict(name="stew",fs=128.,x=np.random.default_rng(8).normal(size=(24,4,32)).astype(np.float32),
        y=np.tile([0,1],12),channels=["F3","F4","C3","C4"],
        meta=pd.DataFrame(dict(subject=np.repeat([1,2,3],8),session=1,task="test",start_sample=np.arange(24)*32)))
    record=dict(mstgc_temporal_hidden=4,mstgc_graph_hidden=4,mstgc_fusion_dim=7,
        mstgc_num_heads=2,mstgc_time_points=8,subspacedims=3,mstgc_dropout=0.)
    domains=np.repeat([1,2,3],8)
    method=f.f5._load_methods(None)[-1]
    model=f.f5._build_model(ds,domains,np.arange(24),np.arange(8),method,f.f5._model_config(record),torch.device("cpu"))
    run=tmp_path/"run"; (run/"subject_03").mkdir(parents=True)
    torch.save(model.state_dict(),run/"subject_03"/"model.pt")
    context=dict(dataset_object=ds,protocol="loso",spec=dict(name="stew"),cache_sha256="fixture")
    info=dict(record=record,summary=pd.DataFrame(dict(subject=[3])),run_dir=str(run))
    args=SimpleNamespace(output_dir=str(tmp_path/"results"),device="cpu",batch_size=4)
    values,audit=f.extract_subject(context,info,method,3,args)
    assert values["pre"].shape==values["post"].shape==(8,4)
    assert audit["state_unchanged"] and audit["passed"]
    cached,again=f.extract_subject(context,info,method,3,args)
    np.testing.assert_array_equal(cached["pre"],values["pre"])
    context["cache_sha256"]="different_data"
    with pytest.raises(ValueError,match="provenance"):
        f.extract_subject(context,info,method,3,args)


@pytest.mark.parametrize("backend",["mne","matplotlib"])
def test_layout_and_composite(tmp_path,backend):
    channels=["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2"]
    xy,source=f.load_layout(channels)
    assert np.isfinite(xy).all() and "MNE" in source
    with pytest.raises(ValueError,match="Unknown electrode"):
        f.load_layout(["invented","F3","F4","Cz"])
    rng=np.random.default_rng(42)
    data=[]
    for name in ("STEW","EEGMAT"):
        pre=rng.normal(size=(5,len(channels)))
        post=rng.normal(size=(5,len(channels)))
        scores=pd.DataFrame(dict(subject_id=[1,2,3,4,5],bacc=[.5,.6,.7,.8,.9]))
        data.append(dict(display_name=name,pre=pre,post=post,xy=xy,subjects=[1,2,3,4,5],
            selection=f.select_representative_subjects(scores)))
    args=SimpleNamespace(output_dir=str(tmp_path),color_scope="dataset",backend=backend)
    f.plot_figure(data,args)
    assert (tmp_path/"Fig6_node_response_topomaps.png").stat().st_size>1000
    assert data[0]["color_limit"]==pytest.approx(np.max(np.abs(np.r_[data[0]["pre"],data[0]["post"]])))
