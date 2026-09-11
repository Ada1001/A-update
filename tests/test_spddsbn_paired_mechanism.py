"""Synthetic numerical fixtures ONLY; never paper experiment observations."""
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest
import torch
from analysis import fig_spddsbn_paired_mechanism_analysis as m


def matrices(n=40):
    rng = np.random.default_rng(12)
    x = rng.normal(size=(n, 3, 3))
    return x @ x.swapaxes(-1, -2) + 2 * np.eye(3)


def test_airm_congruence_and_known_diagonal():
    a, b = np.diag([1., 4., 9.]), np.diag([2., 2., 3.])
    assert m.distances(a,b) == pytest.approx(np.linalg.norm(np.log([2,.5,1/3])))
    x = matrices()
    g = np.array([[1.,.2,0],[0,2,.3],[.1,0,1]])
    np.testing.assert_allclose(m.distances(x[0],x),m.distances(g@x[0]@g.T,g@x@g.T),atol=1e-12)


def test_neighbors_identical_and_pair_sampling():
    x = matrices(20)
    knn,rho,n = m.structure(x,x,k=15,max_pairs=73,block=7)
    assert knn == 1 and rho == pytest.approx(1) and n == 73
    with pytest.raises(ValueError,match="exceed"):
        m.structure(x[:10],x[:10])


def test_fisher_scatter_definition_and_effect_direction():
    x = np.array([[0.],[2.],[4.],[6.]])
    assert m.fisher(x,np.array([0,0,1,1])) == pytest.approx(4.)
    frame = pd.DataFrame({p+suffix: np.arange(1,5,dtype=float)+(0 if suffix=="_pre" else 2)
        for p in ("domain_discrepancy","fisher_source","fisher_target","fisher_pooled") for suffix in ("_pre","_post")})
    result = m.statistics(frame,42)
    assert (result.dropna(subset=["rank_biserial"]).rank_biserial == 1).all()


def test_analyze_common_space_render_cache_and_tamper(tmp_path):
    folder = tmp_path / "folds" / "subject_21"
    folder.mkdir(parents=True)
    x = matrices()
    meta = pd.DataFrame(dict(sample_id=np.arange(40),subject_id=np.repeat([1,21],20),
        fold_id=21,domain=np.repeat(["source","target"],20),true_label=np.tile([0,1],20),
        class_name=np.tile(["Low","High"],20)))
    meta.to_csv(folder/"samples.csv",index=False)
    np.savez_compressed(folder/"paired_spd.npz",pre=x,post=x,**{k:meta[k].to_numpy(dtype=str if k=="domain" else int) for k in ("sample_id","subject_id","fold_id","domain","true_label")})
    audit=dict(passed=True,subject_id=21,archive_sha256=m.diag.digest(folder/"paired_spd.npz"),metadata_sha256=m.diag.digest(folder/"samples.csv"))
    (folder/"audit.json").write_text(json.dumps(audit))
    args = SimpleNamespace(k=15,max_pairs=100,seed=42,distance_block=10,mean_tolerance=1e-7,
                           mean_iterations=200,representative_subject=21,plot_points=40,output_dir=str(tmp_path))
    row=m.analyze_fold(folder,args)
    assert row["knn_source"] == 1
    assert row["domain_discrepancy_pre"] == pytest.approx(row["domain_discrepancy_post"])
    assert row == m.analyze_fold(folder,args)
    coords=np.load(folder/"common_coordinates.npz")
    np.testing.assert_allclose(coords["pre"],coords["post"])
    frame=pd.DataFrame([row])
    m.plot_and_report(frame,m.statistics(frame,42),args)
    assert (tmp_path/"Fig_SPDDSBN_Paired_Mechanism.pdf").stat().st_size > 1000
    with open(folder/"samples.csv","a") as f:
        f.write("tamper")
    with pytest.raises(ValueError,match="audit"):
        m.analyze_fold(folder,args)


def test_real_model_export_contract_and_target_invariance(tmp_path, monkeypatch):
    dataset = {"x": np.random.default_rng(9).normal(size=(24,4,32)).astype(np.float32),
               "y": np.tile([0,1],12), "fs":128.,
               "meta":pd.DataFrame(dict(subject=np.repeat([1,2,3],8),session=1,
                                         task="test",start_sample=np.arange(24)*32))}
    domains=np.repeat([1,2,3],8)
    source,target,val=np.arange(8),np.arange(16,24),np.arange(8,16)
    split=dict(source_ids=source,target_ids=target,val_ids=val,config={"seed":42},
               normalizer=m.f5.fit_source_normalizer(dataset["x"],source))
    record=dict(mstgc_temporal_hidden=4,mstgc_graph_hidden=4,mstgc_fusion_dim=7,
                mstgc_num_heads=2,mstgc_time_points=8,subspacedims=3,mstgc_dropout=0.)
    method=m.f5._load_methods(None)[-1]
    model=m.f5._build_model(dataset,domains,np.arange(24),source,method,m.f5._model_config(record),torch.device("cpu"))
    checkpoint=tmp_path/"run"/"subject_03"/"model.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(model.state_dict(),checkpoint)
    monkeypatch.setattr(m.f5,"_make_split_context",lambda *a:split)
    monkeypatch.setattr(m.f5,"_split_config",lambda *a:split["config"])
    args=SimpleNamespace(output_dir=str(tmp_path/"export"),batch_size=4,device="cpu")
    m.export_fold(dict(dataset_object=dataset,domains=domains),dict(run_dir=str(checkpoint.parent.parent),record=record),method,3,args)
    audit=json.loads((tmp_path/"export"/"folds"/"subject_03"/"audit.json").read_text())
    assert audit["target_predictions_identical"] and audit["checkpoint_unchanged"]
    assert audit["target_logit_max_abs_delta"] == 0
    assert audit["pre_spd"]["shape"] == [16,3,3]
