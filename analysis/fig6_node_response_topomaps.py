"""Real-checkpoint AGMNet node-response topomaps; no training or surrogate data."""
import argparse
import copy
import inspect
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from analysis import fig5_representation_alignment as f5
from analysis.diagnose_fig5_alignment import digest
from src.cl_tsmnet.spd_pca import migrate_legacy_spddsbn_buffers


def load_checkpoints_and_data(spec,args,settings):
    local=copy.copy(args)
    local.output_root=settings.get("output_root",args.output_root)
    local.master_summary=settings.get("master_summary",args.master_summary)
    local.protocol="loso"
    local.allow_missing_master_config=local.allow_legacy_refit=False
    local.target_fs_stew=local.target_fs_eegmat=local.target_fs_cog_bci=None
    if not Path(local.master_summary).is_file():
        raise FileNotFoundError("Missing master summary: "+local.master_summary)
    method=f5._load_methods(None)[-1]
    info=f5._resolve_run(spec,method,local,pd.read_csv(local.master_summary))
    f5._validate_comparable_runs([info],[method],local)
    context=f5._load_dataset_context(spec,local)
    expected=set(context["dataset_object"]["meta"].subject.astype(int))
    requested=settings.get("subjects")
    subjects=sorted(map(int,requested)) if requested is not None else sorted(expected)
    if len(subjects)<4 or len(set(subjects))!=len(subjects):
        raise ValueError("At least four distinct evaluated subjects are needed")
    available=set(info["summary"].subject.astype(int))
    if not set(subjects)<=available or not set(subjects)<=expected:
        raise ValueError("Missing checkpoint summary or data subjects: "+str(set(subjects)-(available&expected)))
    for subject in subjects:
        f5._checkpoint_path(info["run_dir"],subject)
    context["cache_sha256"]=digest(context["cache"])
    return context,info,method,subjects


def extract_pre_post_node_features(model,windows,domains):
    """Capture graph input/output from the SAME classifier forward.

    [N,C,F,T] is flattened to [N,C,F*T], without averaging over time.
    POST is before channel reliability weighting and SPD construction.
    """
    if getattr(model,"graph",None) is None:
        raise ValueError("This analysis requires a trained electrode graph module")
    captured={}
    def before(module,inputs): captured["pre"]=inputs[0].detach()
    def after(module,inputs,output): captured["post"]=output.detach()
    hooks=[model.graph.register_forward_pre_hook(before),model.graph.register_forward_hook(after)]
    try:
        with torch.no_grad():
            logits=model(windows,domains)
    finally:
        for hook in hooks: hook.remove()
    if set(captured)!={"pre","post"}:
        raise RuntimeError("Graph hooks did not capture both feature locations")
    pre,post=captured["pre"],captured["post"]
    if pre.shape!=post.shape or pre.ndim not in (3,4):
        raise ValueError("Matched node dimensions required for direct L2 comparison: {} vs {}".format(pre.shape,post.shape))
    return pre.flatten(2),post.flatten(2),logits


def compute_node_response(features):
    values=features.detach().cpu().double().numpy() if torch.is_tensor(features) else np.asarray(features,dtype=np.float64)
    if values.ndim!=3 or not np.isfinite(values).all():
        raise ValueError("Expected finite [N,C,D] node features")
    return np.linalg.norm(values,axis=-1)


def compute_subject_contrast_topomap(responses,labels,high_label):
    classes=np.unique(labels)
    if len(classes)!=2 or high_label not in classes:
        raise ValueError("Exactly two classes including the configured positive class are required")
    low_label=next(c for c in classes if c!=high_label)
    low=np.asarray(responses)[labels==low_label].mean(0)
    high=np.asarray(responses)[labels==high_label].mean(0)
    return dict(low=low,high=high,delta=high-low,low_label=int(low_label),high_label=int(high_label))


def select_representative_subjects(scores,explicit=None):
    scores=scores.sort_values("subject_id").reset_index(drop=True)
    if explicit is not None:
        if len(explicit)!=4 or len(set(explicit))!=4 or not set(explicit)<=set(scores.subject_id):
            raise ValueError("Representative list must contain four unique evaluated subjects")
        return [dict(subject_id=int(s),criterion="user_fixed",bacc=float(scores.loc[scores.subject_id==s,"bacc"].iloc[0])) for s in explicit]
    chosen=[]
    targets=[("q25",scores.bacc.quantile(.25)),("median",scores.bacc.median()),
             ("q75",scores.bacc.quantile(.75)),("mean",scores.bacc.mean())]
    for criterion,target in targets:
        candidates=scores[~scores.subject_id.isin([r["subject_id"] for r in chosen])].copy()
        candidates["distance"]=(candidates.bacc-target).abs()
        row=candidates.sort_values(["distance","subject_id"]).iloc[0]
        chosen.append(dict(subject_id=int(row.subject_id),criterion=criterion,target_bacc=float(target),bacc=float(row.bacc)))
    return chosen


def compute_group_average_topomap(contrasts):
    return np.asarray(contrasts,dtype=float).mean(axis=0)  # equal subject weights


def compute_topomap_consistency(contrasts,pair_mask=None):
    values=np.asarray(contrasts,dtype=np.float64)
    centered=values-values.mean(1,keepdims=True)
    norms=np.linalg.norm(centered,axis=1)
    valid=norms>np.finfo(float).eps*max(1.,float(np.max(np.abs(values))))*values.shape[1]
    normalized=np.divide(centered,norms[:,None],out=np.zeros_like(centered),where=valid[:,None])
    i,j=np.triu_indices(len(values),1)
    correlations=np.clip(np.sum(normalized[i]*normalized[j],axis=1),-1,1)
    mask=valid[i]&valid[j]
    if pair_mask is not None: mask &= pair_mask
    correlations[~mask]=np.nan
    return dict(mean=float(correlations[mask].mean()) if mask.any() else np.nan,
        correlations=correlations,mask=mask,pairs=list(zip(i.tolist(),j.tolist())),
        valid_pairs=int(mask.sum()),constant_subjects=np.flatnonzero(~valid).tolist())


def extract_subject(context,info,method,subject,args):
    ds=context["dataset_object"]
    split=f5._make_split_context(context,subject,f5._split_config(info))
    checkpoint=f5._checkpoint_path(info["run_dir"],subject)
    sha=digest(checkpoint)
    folder=Path(args.output_dir)/"node_response_cache"/context["spec"]["name"]/("subject_%02d"%subject)
    folder.mkdir(parents=True,exist_ok=True)
    signature=dict(schema=1,checkpoint_sha256=sha,cache_sha256=context["cache_sha256"],
        model_config=f5._model_config(info["record"]),split=split["config"],device=args.device,batch_size=args.batch_size)
    audit_path=folder/"audit.json"; data_path=folder/"responses.npz"
    if audit_path.exists() and data_path.exists():
        audit=json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("signature")==signature and audit.get("response_sha256")==digest(data_path):
            with np.load(data_path,allow_pickle=False) as z: return {k:z[k] for k in z.files},audit
        raise ValueError("Cache provenance mismatch; use a fresh output directory: "+str(folder))
    ids=split["target_ids"]
    selected=np.concatenate([split["source_ids"],split["val_ids"],ids])
    domains=split["domains"]
    model=f5._build_model(ds,domains,selected,split["source_ids"],method,
        f5._model_config(info["record"]),torch.device(args.device))
    state,migrations=migrate_legacy_spddsbn_buffers(f5._load_state(checkpoint),model.state_dict())
    model.load_state_dict(state,strict=True); model.eval()
    snapshot={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    responses={"pre":[],"post":[],"logits":[]}
    dimension=None
    for start in range(0,len(ids),args.batch_size):
        ix=ids[start:start+args.batch_size]
        x=torch.from_numpy(split["normalizer"].transform_array(ds["x"][ix])).to(args.device)
        d=torch.from_numpy(domains[ix]).long().to(args.device)
        pre,post,logits=extract_pre_post_node_features(model,x,d)
        dimension=int(pre.shape[-1])
        responses["pre"].append(compute_node_response(pre))
        responses["post"].append(compute_node_response(post))
        responses["logits"].append(logits.detach().cpu().numpy())
    values={k:np.concatenate(v) for k,v in responses.items()}
    if any(not np.isfinite(v).all() for v in values.values()): raise ValueError("Non-finite model output")
    if any(not torch.equal(v.detach().cpu(),snapshot[k]) for k,v in model.state_dict().items()):
        raise RuntimeError("Extraction changed model state")
    if digest(checkpoint)!=sha: raise RuntimeError("Checkpoint changed during extraction")
    channels=np.asarray(ds["channels"],dtype=str)
    if values["pre"].shape!=(len(ids),len(channels)): raise ValueError("Node/electrode alignment mismatch")
    values.update(sample_id=ids,subject_id=ds["meta"].iloc[ids].subject.to_numpy(int),
                  label=ds["y"][ids],channels=channels)
    if len(np.unique(ids))!=len(ids) or not np.all(values["subject_id"]==subject):
        raise ValueError("LOSO target identity mismatch")
    bacc=float(balanced_accuracy_score(values["label"],values["logits"].argmax(1)))
    audit=dict(signature=signature,bacc=bacc,subject_id=subject,passed=True,migrations=migrations,
        n_samples=len(ids),feature_dimension=dimension,checkpoint=str(checkpoint),
        response_definition="L2 over flattened feature and time axes",state_unchanged=True)
    summary=info["summary"].loc[info["summary"].subject.astype(int)==subject]
    if len(summary)!=1: raise ValueError("Expected one summary row per LOSO subject")
    if "test_bacc" in summary:
        audit["reported_bacc"]=float(summary.test_bacc.iloc[0])
        audit["bacc_delta_vs_summary"]=bacc-audit["reported_bacc"]
    np.savez_compressed(data_path,**values)
    audit["response_sha256"]=digest(data_path)
    f5._write_json(audit,str(audit_path))
    return values,audit


def load_layout(channels,csv_path=None):
    """Never invent electrode positions when the montage is unavailable."""
    if csv_path:
        frame=pd.read_csv(csv_path)
        if frame.channel.duplicated().any(): raise ValueError("Duplicate layout channels")
        frame=frame.set_index("channel").reindex(channels)
        xy=frame[["x","y"]].to_numpy(float)
        source="provided scalp projection (x=right, y=anterior): "+str(csv_path)
    else:
        try:
            import mne
        except ImportError as exc:
            raise ImportError("Install mne or supply --run-config layout CSV; no guessed electrodes are used") from exc
        montage=mne.channels.make_standard_montage("standard_1020")
        names={n.casefold():n for n in montage.ch_names}
        missing=[n for n in channels if n.casefold() not in names]
        if missing: raise ValueError("Unknown electrode names; supply a measured layout CSV: "+str(missing))
        info=mne.create_info([names[n.casefold()] for n in channels],128.,"eeg")
        info.set_montage(montage)
        from mne.channels.layout import _find_topomap_coords
        xy=_find_topomap_coords(info,np.arange(len(channels)),sphere=(0.,0.,0.,.095))
        source="MNE standard_1020 projected electrode coordinates"
    if not np.isfinite(xy).all() or len(np.unique(xy,axis=0))!=len(channels):
        raise ValueError("Missing/nonfinite/duplicate electrode coordinates")
    if len(xy)<4: raise ValueError("At least four electrodes required")
    # Isotropic change of display units; relative positions are preserved.
    xy=xy/max(np.linalg.norm(xy,axis=1).max(),1e-12)*.9
    return xy,source


def draw_topomap(ax,values,xy,limit,backend="auto"):
    import matplotlib.pyplot as plt
    try:
        if backend=="matplotlib": raise ImportError()
        import mne
    except ImportError:
        if backend=="mne": raise
        from scipy.interpolate import griddata
        gx,gy=np.meshgrid(np.linspace(-1,1,160),np.linspace(-1,1,160))
        z=griddata(xy,values,(gx,gy),method="linear")
        z[gx*gx+gy*gy>1]=np.nan
        image=ax.imshow(z,extent=(-1,1,-1,1),origin="lower",cmap="RdBu_r",vmin=-limit,vmax=limit)
        ax.add_patch(plt.Circle((0,0),1,fill=False,color="black",lw=.5))
        ax.plot([-.1,0,.1],[.99,1.10,.99],color="black",lw=.5)
        ax.scatter(*xy.T,s=2,c="black"); ax.set(xlim=(-1.12,1.12),ylim=(-1.08,1.15),aspect="equal")
        ax.axis("off")
        return image
    options=dict(axes=ax,show=False,cmap="RdBu_r",contours=3,sensors=True,
                 sphere=(0.,0.,0.,1.),extrapolate="local",res=256)
    if "vlim" in inspect.signature(mne.viz.plot_topomap).parameters: options["vlim"]=(-limit,limit)
    else: options.update(vmin=-limit,vmax=limit)
    image,_=mne.viz.plot_topomap(values,xy,**options)
    for line in ax.lines: line.set_linewidth(.6)
    return image


def plot_group_level_topomaps(axes,pre,post,xy,limit,backend):
    draw_topomap(axes[0],compute_group_average_topomap(pre),xy,limit,backend)
    draw_topomap(axes[1],compute_group_average_topomap(post),xy,limit,backend)
    draw_topomap(axes[2],np.sign(post).mean(0),xy,1.,backend)
    for ax,title in zip(axes,("Pre-graph","Post-graph","Sign consistency")): ax.set_title(title,fontsize=6.5,pad=2)


def plot_representative_subjects(axes,post,subjects,selection,xy,limit,backend):
    for ax,row in zip(axes[:4],selection):
        draw_topomap(ax,post[subjects.index(row["subject_id"])],xy,limit,backend)
        ax.set_title("S%02d"%row["subject_id"],fontsize=6.5,pad=2)
    draw_topomap(axes[4],compute_group_average_topomap(post),xy,limit,backend)
    axes[4].set_title("Avg",fontsize=6.5,pad=2)


def plot_figure(datasets,args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":6.5,"pdf.fonttype":42,"ps.fonttype":42})
    fig=plt.figure(figsize=(7.16,1.35*len(datasets)+.65))
    grid=fig.add_gridspec(len(datasets),9,width_ratios=[1,1,1,.22,1,1,1,1,1],
                        left=.055,right=.985,top=.82,bottom=.19,hspace=.65,wspace=.06)
    global_limit=max(float(np.max(np.abs(np.r_[d["pre"],d["post"]]))) for d in datasets)
    for i,data in enumerate(datasets):
        axes=[fig.add_subplot(grid[i,j]) for j in (0,1,2,4,5,6,7,8)]
        limit=max(global_limit if args.color_scope=="global" else float(np.max(np.abs(np.r_[data["pre"],data["post"]]))),1e-12)
        data["color_limit"]=limit
        plot_group_level_topomaps(axes[:3],data["pre"],data["post"],data["xy"],limit,args.backend)
        plot_representative_subjects(axes[3:],data["post"],data["subjects"],data["selection"],data["xy"],limit,args.backend)
        pos=axes[0].get_position()
        fig.text(.009,pos.y0+pos.height/2,data["display_name"],rotation=90,va="center",weight="bold",fontsize=8)
        # Different units: contrast shares one scale; signed agreement has its own.
        cax=fig.add_axes([pos.x0,pos.y0-.055,.20,.012])
        cb=fig.colorbar(ScalarMappable(norm=Normalize(-limit,limit),cmap="RdBu_r"),cax=cax,orientation="horizontal")
        cb.set_label("High − Low response (a.u.)",fontsize=6); cb.ax.tick_params(labelsize=5.5,pad=1)
        pos3=axes[2].get_position()
        cax2=fig.add_axes([pos3.x0,pos.y0-.055,pos3.width,.012])
        cb2=fig.colorbar(ScalarMappable(norm=Normalize(-1,1),cmap="RdBu_r"),cax=cax2,orientation="horizontal",ticks=[-1,0,1])
        cb2.set_label("Mean sign",fontsize=6); cb2.ax.tick_params(labelsize=5.5,pad=1)
    fig.text(.05,.92,"(A) Group-level spatial response maps",weight="bold",fontsize=8)
    fig.text(.44,.92,"(B) Representative subject examples",weight="bold",fontsize=8)
    fig.text(.5,.99,"Learned spatial response patterns before and after graph propagation",ha="center",va="top",fontsize=8.5)
    fig.text(.5,.025,"Held-out LOSO subjects • Equal subject weighting • Four BAcc-based examples and group mean",ha="center",fontsize=6)
    for ext in ("pdf","png"):
        fig.savefig(Path(args.output_dir)/("Fig6_node_response_topomaps."+ext),dpi=600)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets",default="stew,eegmat"); p.add_argument("--dataset-labels",default="STEW,EEGMAT")
    p.add_argument("--output-root",default="outputs"); p.add_argument("--master-summary",default="outputs/master_summary.csv")
    p.add_argument("--data-root",default="data"); p.add_argument("--cache-root",default="outputs/cache")
    p.add_argument("--output-dir",default="results"); p.add_argument("--run-config",help="JSON dataset -> output_root,master_summary,layout,representatives,subjects,high_label")
    p.add_argument("--batch-size",type=int,default=16); p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--seed",type=int,default=42); p.add_argument("--threads",type=int,default=1)
    p.add_argument("--backend",choices=["auto","mne","matplotlib"],default="auto")
    p.add_argument("--color-scope",choices=["dataset","global"],default="dataset")
    args=p.parse_args()
    if args.batch_size<1 or args.threads<1: p.error("batch-size and threads must be positive")
    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(args.threads)
    settings=json.loads(Path(args.run_config).read_text(encoding="utf-8")) if args.run_config else {}
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    data_rows=[]; consistency=[]; selections=[]; contrast_rows=[]; all_scores=[]; pair_rows=[]
    summaries=["# Learned spatial response patterns before and after graph propagation",
        "Only trained full AGMNet checkpoints and held-out target windows are analyzed. Graph input/output are captured from the same eval forward; weights and buffers remain unchanged.",
        "Responses are L2 norms over flattened feature×time axes, not raw EEG amplitude. POST is before channel reliability weighting. The graph block also includes learned projection and ReLU; this comparison does not isolate adjacency propagation alone.",
        "Across-subject maps come from separately trained LOSO models. Equal subject weighting is used. Response scale can differ across checkpoints; no separate PRE/POST normalization is applied. Consistency refers to spatial correlation, not temporal stability."]
    for spec in f5._parse_datasets(args.datasets,args.dataset_labels):
        conf=settings.get(spec["name"],{})
        context,info,method,subjects=load_checkpoints_and_data(spec,args,conf)
        ds=context["dataset_object"]
        classes=np.unique(ds["y"])
        if len(classes)!=2: raise ValueError("Binary task required: "+spec["name"])
        names={int(c):str(ds.get("label_names",{}).get(int(c),c)) for c in classes}
        named_high=[c for c in classes if "high" in names[int(c)].lower()]
        high=int(conf.get("high_label",named_high[0] if len(named_high)==1 else classes[-1]))
        if high not in classes: raise ValueError("Configured high_label absent")
        low=int(next(c for c in classes if c!=high))
        pre=[]; post=[]; scores=[]
        for subject in subjects:
            print("Extracting",spec["name"],subject,flush=True)
            values,audit=extract_subject(context,info,method,subject,args)
            if not np.array_equal(values["channels"],np.asarray(ds["channels"],dtype=str)):
                raise ValueError("Channel order differs across subjects")
            scores.append(dict(dataset=spec["name"],subject_id=subject,bacc=audit["bacc"],
                reported_bacc=audit.get("reported_bacc"),bacc_delta_vs_summary=audit.get("bacc_delta_vs_summary")))
            for phase,destination in (("pre",pre),("post",post)):
                result=compute_subject_contrast_topomap(values[phase],values["label"],high)
                destination.append(result["delta"])
                for i,ch in enumerate(values["channels"]):
                    contrast_rows.append(dict(dataset=spec["name"],subject_id=subject,channel=ch,phase=phase,
                        low_response=result["low"][i],high_response=result["high"][i],contrast=result["delta"][i]))
        pre,post=np.stack(pre),np.stack(post)
        selection=select_representative_subjects(pd.DataFrame(scores),conf.get("representatives"))
        selections.extend(dict(dataset=spec["name"],**r) for r in selection); all_scores.extend(scores)
        mask=compute_topomap_consistency(pre)["mask"]&compute_topomap_consistency(post)["mask"]
        cpre,cpost=(compute_topomap_consistency(v,mask) for v in (pre,post))
        consistency.append(dict(dataset=spec["name"],n_subjects=len(subjects),total_pairs=len(mask),valid_common_pairs=int(mask.sum()),
            consistency_pre=cpre["mean"],consistency_post=cpost["mean"],change=cpost["mean"]-cpre["mean"]))
        for pair,vpre,vpost in zip(cpre["pairs"],cpre["correlations"],cpost["correlations"]):
            pair_rows.append(dict(dataset=spec["name"],subject_a=subjects[pair[0]],subject_b=subjects[pair[1]],pre=vpre,post=vpost))
        xy,layout_source=load_layout(list(ds["channels"]),conf.get("layout"))
        pd.DataFrame(dict(channel=ds["channels"],x=xy[:,0],y=xy[:,1],source=layout_source)).to_csv(out/(spec["name"]+"_layout.csv"),index=False)
        data_rows.append(dict(display_name=spec["display_name"],pre=pre,post=post,xy=xy,subjects=subjects,selection=selection))
        print(spec["display_name"],"pre consistency",cpre["mean"],"post consistency",cpost["mean"],flush=True)
        direction=("Post-graph spatial response patterns were more consistent across the analyzed subjects (descriptive correlation increase)." if cpost["mean"]>cpre["mean"] else
            "Spatial consistency did not increase; these results do not support an improvement claim.") if mask.any() else "Consistency is undefined because no common valid pairs remain."
        summaries.extend(["## "+spec["display_name"],
            "Contrast: {} (label {}) minus {} (label {}). If these are not workload levels, High/Low are plotting aliases only.".format(names[high],high,names[low],low),
            "Subjects: {}; PRE mean Pearson r={:.6g}; POST={:.6g}; valid common pairs {}/{}. {}".format(len(subjects),cpre["mean"],cpost["mean"],mask.sum(),len(mask),direction),
            "Representatives: "+", ".join("S{:02d} ({}, BAcc={:.4f})".format(r["subject_id"],r["criterion"],r["bacc"]) for r in selection),
            "Layout: "+layout_source])
    plot_figure(data_rows,args)
    pd.DataFrame(consistency).to_csv(out/"fig6_topomap_consistency.csv",index=False)
    pd.DataFrame(selections).to_csv(out/"fig6_subject_selection.csv",index=False)
    pd.DataFrame(contrast_rows).to_csv(out/"fig6_subject_contrasts.csv",index=False)
    pd.DataFrame(all_scores).to_csv(out/"fig6_subject_bacc.csv",index=False)
    pd.DataFrame(pair_rows).to_csv(out/"fig6_pairwise_correlations.csv",index=False)
    summaries.extend(["Raw contrast color limits (symmetric): "+str({d["display_name"]:d["color_limit"] for d in data_rows}),
        "The third group panel is mean(sign(POST contrast)) across subjects, range [-1,1], with its own colorbar. Zero contrasts contribute zero. This is sign agreement, not Pearson correlation or significance.",
        "The examples are selected by BAcc, not by map appearance. Pairwise correlations are dependent and no significance test is claimed. Class-conditioned contrasts are descriptive and do not establish decoding accuracy or repeatability. Changes in concentration have not been separately quantified."])
    (out/"fig6_summary.md").write_text("\n\n".join(summaries),encoding="utf-8")
    f5._write_json(dict(arguments=vars(args),dataset_settings=settings),str(out/"fig6_provenance.json"))


if __name__=="__main__":
    main()
