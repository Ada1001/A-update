import numpy as np
import pytest
from analysis.fig6_reporting import paired_subject_analysis


def test_paired_bootstrap_identical_maps_zero_change():
    x=np.random.default_rng(1).normal(size=(8,14))
    summary, rows, draws=paired_subject_analysis(x,x,list(range(8)),200,42)
    np.testing.assert_array_equal(draws,0)
    assert summary['change']==summary['conditional_bootstrap_low']==summary['conditional_bootstrap_high']==0
    assert summary['loo_min']==summary['loo_max']==0
    assert rows.valid_partners.tolist()==[7]*8


def test_bootstrap_preserves_pairing_and_excludes_self_pairs():
    rng=np.random.default_rng(8)
    pre=rng.normal(size=(6,10)); post=rng.normal(size=(6,10))
    result,rows,draws=paired_subject_analysis(pre,post,list(range(6)),100,42)
    delta=np.corrcoef(post)-np.corrcoef(pre)
    tri=np.triu_indices(6,1)
    weights=np.bincount(np.random.default_rng(42).integers(0,6,6),minlength=6)
    w=weights[tri[0]]*weights[tri[1]]
    assert draws[0]==pytest.approx(np.dot(w,delta[tri])/w.sum())
    assert result['change']==pytest.approx(delta[tri].mean())
    assert rows.iloc[0].leave_one_out_change==pytest.approx(delta[1:,1:][np.triu_indices(5,1)].mean())


def test_common_valid_pairs_and_repeatability():
    rng=np.random.default_rng(6)
    pre=rng.normal(size=(8,14)); post=rng.normal(size=(8,14)); pre[0]=1
    a,rows,x=paired_subject_analysis(pre,post,list(range(8)),200,42)
    b,_,y=paired_subject_analysis(pre,post,list(range(8)),200,42)
    assert a==b and a['valid_pairs']==21
    assert rows.iloc[0].valid_partners==0 and np.isnan(rows.iloc[0].change)
    np.testing.assert_array_equal(x,y)
    with pytest.raises(ValueError,match='No common valid'):
        paired_subject_analysis(np.ones((8,14)),post,list(range(8)),200,42)


def test_three_rows_shared_class_ranges_and_display_scope(tmp_path):
    from types import SimpleNamespace
    import pandas as pd
    from analysis.fig6_reporting import plot_report
    from analysis.fig6_node_response_topomaps import load_layout
    rng=np.random.default_rng(17)
    channels=['F3','F4','C3','C4','P3','P4','O1','O2']
    xy,_=load_layout(channels)
    low=rng.uniform(20,30,(5,8)); high=low+rng.normal(size=(5,8))
    post_low=low+10; post_high=post_low+rng.normal(size=(5,8))
    post_high[4,0]+=100  # omitted representative still contributes to group Avg
    d=dict(display_name='TEST',subjects=[1,2,3,4,5],xy=xy,
           selection=[dict(subject_id=s) for s in [1,2,3,4]],
           pre=high-low,post=post_high-post_low,pre_high=high,pre_low=low,
           post_high=post_high,post_low=post_low)
    args=SimpleNamespace(output_dir=str(tmp_path),layout='ab-three-rows',
                         color_scope='dataset',color_mode='group-detail',backend='matplotlib',
                         bootstrap_replicates=100,seed=42)
    plot_report([d],args)
    limits=pd.read_csv(tmp_path/'fig6_color_limits.csv')
    assert len(limits)==6
    for panel in ['A','B']:
        rows=limits[(limits.panel==panel)&(limits.row!='High - Low')]
        assert rows.vmin.nunique()==rows.vmax.nunique()==1
    expected=max(np.abs(d['post'][:4]).max(),np.abs(d['post'].mean(0)).max())
    assert limits[(limits.panel=='B')&(limits.row=='High - Low')].vmax.iloc[0]==pytest.approx(expected)
    assert (tmp_path/'Fig6_node_response_topomaps.pdf').stat().st_size>1000
    assert (tmp_path/'Fig6_all_subjects_test_post_contrast.png').stat().st_size>1000
    atlas_limits=pd.read_csv(tmp_path/'fig6_all_subjects_color_limits.csv')
    assert atlas_limits.vmax.iloc[0]==10
    assert atlas_limits.vmin.iloc[0]==-10
    assert atlas_limits.above_limit.iloc[0]==int((d['post']>10).sum())
    assert atlas_limits.n_subjects.iloc[0]==5
    d['pre_high']=high+1
    with pytest.raises(ValueError,match='High-Low'):
        plot_report([d],args)
