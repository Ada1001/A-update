"""Synthetic smoke tests only; no experiment findings are generated here."""
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest
import torch
from src.cl_tsmnet.tsmnet_backbone_ablation import TSMBackboneAblation, VARIANTS
from analysis import tsmnet_backbone_ablation as runner


def test_identical_author_backbone_initialization():
    states=[]
    for variant in VARIANTS:
        torch.manual_seed(42)
        m=TSMBackboneAblation(variant,4,32,2,[1,2,3],spatial_filters=6,subspacedims=3)
        states.append((m.original.cnn if variant=="tsmnet" else m.cnn).state_dict())
    for state in states[1:]:
        for key in states[0]:
            torch.testing.assert_close(state[key],states[0][key],rtol=0,atol=0)


def test_domain_batches_cover_samples_without_singletons():
    domains=torch.tensor([1]*17+[2]*33)
    batches=runner.training_batches(domains,16,42)
    assert torch.equal(torch.cat(batches).sort().values,torch.arange(50))
    assert all(len(b)>1 and len(domains[b].unique())==1 for b in batches)
    assert all(torch.equal(a,b) for a,b in zip(batches,runner.training_batches(domains,16,42)))


def test_train_export_and_figures(tmp_path):
    torch.set_num_threads(1)
    ds=dict(name="stew",fs=128.,x=np.random.default_rng(4).normal(size=(100,4,32)).astype(np.float32),
        y=np.repeat([0,1],50),meta=pd.DataFrame(dict(subject=21,session=1,paradigm="test",
                task=np.repeat(["low","high"],50),start_sample=np.tile(np.arange(50)*32,2))))
    context=dict(dataset_object=ds,protocol="single_session")
    args=SimpleNamespace(output_dir=str(tmp_path),protocol="single_session",seed=42,
        single_val_size=.125,val_size=.2,test_size=.2,spatial_filters=6,subspacedims=3,
        lr=.001,weight_decay=.0001,epochs=2,patience=2,batch_size=16,max_points=30,
        fourth_model="tsmnet",representative_subject=21,reducer="tsne")
    for variant in VARIANTS:
        scores=runner.train_fold(context,21,variant,args)
        assert np.isfinite(scores["best_val_loss"])
        folder=tmp_path/variant/"folds"/"subject_21"
        z,meta=runner.load_export(folder)
        assert len(meta)==90
        audit=json.loads((folder/"audit.json").read_text())
        assert audit["target_predictions_identical"]
        restored=TSMBackboneAblation(**audit["model_config"])
        restored.load_state_dict(torch.load(folder/"model.pt",weights_only=True),strict=True)
        if variant=="tsmnet":
            cfg=SimpleNamespace(mean_tolerance=1e-7,mean_iterations=200,k=15,max_pairs=200,
                seed=42,distance_block=128)
            metric=runner.f4.analyze_fold(folder,cfg)
            assert np.isfinite(metric["domain_discrepancy_post"])
        if variant=="mean-eudsbn":
            assert not restored.eudsbn.layers["210011"].affine
            assert restored.affine_weight.requires_grad
    runner.plot_fig5([21],args)
    assert (tmp_path/"fig5_tsmnet"/"fig5_tsmnet_backbone.pdf").exists()
    with pytest.raises(FileExistsError):
        runner.train_fold(context,21,"tsmnet",args)
    args.resume=True
    before=runner.digest(tmp_path/'tsmnet'/'folds'/'subject_21'/'model.pt')
    assert runner.train_fold(context,21,'tsmnet',args)['variant']=='tsmnet'
    assert runner.digest(tmp_path/'tsmnet'/'folds'/'subject_21'/'model.pt')==before
    args.lr=.02
    with pytest.raises(ValueError,match='configuration mismatch'):
        runner.train_fold(context,21,'tsmnet',args)


def test_target_batch_boundaries_are_preserved():
    class BatchSensitive(torch.nn.Module):
        def forward(self,x,d,intermediates):
            # Deterministic fixture makes changes in batch composition observable.
            logits=x+x.mean(0)
            return logits,dict(features=x)
    model=BatchSensitive()
    x=torch.arange(14.).reshape(7,2); tx=torch.arange(10.).reshape(5,2)
    d=torch.ones(7,dtype=torch.long); td=torch.full((5,),2,dtype=torch.long)
    baseline=runner.extract(model,tx,td,4)['logits']
    old=runner.extract(model,torch.cat([x,tx]),torch.cat([d,td]),4)['logits'][7:]
    assert not np.array_equal(old,baseline)
    fixed=runner.extract_partitions(model,x,d,tx,td,4)
    np.testing.assert_array_equal(fixed['logits'][7:],baseline)
    np.testing.assert_array_equal(fixed['features'],torch.cat([x,tx]).numpy())


def test_incomplete_fold_is_archived_not_overwritten(tmp_path):
    folder=tmp_path/'subject_07';folder.mkdir();(folder/'model.pt').write_bytes(b'preserve')
    args=SimpleNamespace(restart_incomplete=False)
    with pytest.raises(ValueError,match='Incomplete fold'):runner.resume_fold(folder,args)
    args.restart_incomplete=True
    assert runner.resume_fold(folder,args) is None
    backups=list(tmp_path.glob('subject_07.incomplete-*'))
    assert len(backups)==1 and (backups[0]/'model.pt').read_bytes()==b'preserve'
    assert not folder.exists()
