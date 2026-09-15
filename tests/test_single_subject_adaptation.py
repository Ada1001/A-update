"""Numerical smoke tests, not scientific experiment results."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from src.cl_tsmnet.splits import make_split, split_domain_ids
from src.cl_tsmnet.training import train_one_split
from analysis import fig5_representation_alignment as f5
from analysis import fig_spddsbn_paired_mechanism_analysis as f4


def data():
    rng=np.random.default_rng(42)
    return dict(name="stew",fs=128.,x=rng.normal(size=(100,4,32)).astype(np.float32),
        y=np.repeat([0,1],50),meta=pd.DataFrame(dict(subject=21,session=1,paradigm="test",
            task=np.repeat(["low","high"],50),start_sample=np.tile(np.arange(50)*32,2))))


def test_single_domains_and_visualization_split_agree():
    ds=data()
    split=make_split(ds,"single_session",21,val_size=.125)
    domains=split_domain_ids(ds,"single_session",split)
    assert [np.unique(domains[split[k]]).tolist() for k in ("train","val","test")] == [[210011],[210012],[210013]]
    ctx=f5._make_split_context(dict(dataset_object=ds,protocol="single_session"),21,
                              dict(seed=42,val_size=.125,test_size=.2,artifact_z=None))
    np.testing.assert_array_equal(ctx["domains"],domains)
    for key,part in (("source_ids","train"),("val_ids","val"),("target_ids","test")):
        np.testing.assert_array_equal(ctx[key],split[part])


@pytest.mark.parametrize("model",["tsmnet","ms_tgc_spddsbn"])
def test_single_training_adaptation_and_checkpoint(tmp_path,model):
    ds=data()
    split=make_split(ds,"single_session",21,val_size=.125)
    result=train_one_split(ds,split_domain_ids(ds,"single_session",split),split,
        str(Path(__file__).resolve().parents[1]),output_dir=str(tmp_path/"subject_21"),model_type=model,
        epochs=2,batch_size=16,refit_batch_size=8,target_adapt=True,augment=False,
        subspacedims=3,spatial_filters=6,temp_kernel=5,
        mstgc_temporal_hidden=4,mstgc_graph_hidden=4,mstgc_fusion_dim=7,
        mstgc_num_heads=2,mstgc_time_points=8,mstgc_dropout=0.)
    assert result["target_adapt"] and result["target_refit_scope"] == "target_only"
    assert result["val_stat_refit"]
    assert np.isfinite(result["test"]["loss"])
    assert (tmp_path/"subject_21"/"model.pt").exists()
    record=dict(protocol="single_session",single_val_size=.125,val_size=.2,test_size=.2,
        mstgc_temporal_hidden=4,mstgc_graph_hidden=4,mstgc_fusion_dim=7,mstgc_num_heads=2,
        mstgc_time_points=8,mstgc_dropout=0.,subspacedims=3,spatial_filters=6,temp_kernel=5)
    context=dict(dataset_object=ds,protocol="single_session",domains=split_domain_ids(ds,"single_session",split),
                 spec=dict(name="stew"),cache="synthetic_test_only")
    info=dict(record=record,summary=pd.DataFrame([dict(subject=21)]),run_dir=str(tmp_path))
    method=f5._load_methods(None,fourth_model="tsmnet" if model=="tsmnet" else None)[-1]
    args=SimpleNamespace(output_dir=str(tmp_path/"fig4"),batch_size=16,device="cpu")
    f4.export_fold(context,info,method,21,args)
    assert (tmp_path/"fig4"/"folds"/"subject_21"/"audit.json").exists()
    import torch
    args = SimpleNamespace(feature_cache_dir=str(tmp_path/"fig5_cache"),force_reextract=False,
        source_calibration="refit",feature_location="representation",batch_size=16,
        device_object=torch.device("cpu"))
    split_context=f5._make_split_context(context,21,f5._split_config(info))
    features, metadata, location=f5.load_feature_set(context,method,info,21,split_context,args)
    assert features.shape[0] == len(split["train"])+len(split["test"])
    assert set(metadata.domain_id)=={210011,210013}
    assert location["calibration_audit"]["target_prediction_identical"]


def test_tsmnet_default_spd_dimension_momentum_training(tmp_path):
    ds=data()
    split=make_split(ds,"single_session",21,val_size=.125)
    result=train_one_split(ds,split_domain_ids(ds,"single_session",split),split,
        str(Path(__file__).resolve().parents[1]),output_dir=str(tmp_path),model_type="tsmnet",
        epochs=12,patience=12,batch_size=16,refit_batch_size=16,target_adapt=True,
        augment=False,tsmnet_bn_schedule="momentum")
    assert result["epochs_ran"]==12
    assert np.isfinite(result["test"]["loss"])
