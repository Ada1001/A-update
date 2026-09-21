"""Independent training and Fig4/Fig5 for TSMNet backbone ablations."""
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from src.cl_tsmnet.tsmnet_backbone_ablation import TSMBackboneAblation, VARIANTS
from src.cl_tsmnet.training import make_optimizer
from src.cl_tsmnet.spd_pca import validate_spd_matrices
from analysis import fig5_representation_alignment as f5
from analysis import fig_spddsbn_paired_mechanism_analysis as f4
from analysis.diagnose_fig5_alignment import digest


def extract(model,x,d,batch_size):
    model.eval()
    parts={}
    with torch.no_grad():
        for start in range(0,len(x),batch_size):
            logits, values=model(x[start:start+batch_size],d[start:start+batch_size],True)
            for k,v in dict(logits=logits,**values).items():
                parts.setdefault(k,[]).append(v.detach().cpu().numpy())
    result={k:np.concatenate(v) for k,v in parts.items()}
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError("Non-finite exported features")
    return result


def training_batches(domains,batch_size,seed):
    """One domain per batch, same label-blind ordering for every variant.

    Merge singleton tails to avoid a degenerate domain-statistics estimate.
    """
    if batch_size<2:
        raise ValueError("Domain BN training requires batch_size >= 2")
    generator=torch.Generator().manual_seed(seed)
    batches=[]
    for domain in domains.unique():
        ids=torch.where(domains==domain)[0]
        if len(ids)<2:
            raise ValueError("Each source domain needs at least two windows")
        ids=ids[torch.randperm(len(ids),generator=generator)]
        chunks=list(ids.split(batch_size))
        if len(chunks)>1 and len(chunks[-1])==1:
            chunks[-2]=torch.cat(chunks[-2:]); chunks.pop()
        batches.extend(chunks)
    return [batches[i] for i in torch.randperm(len(batches),generator=generator).tolist()]


def train_fold(context,subject,variant,args):
    out=Path(args.output_dir)/variant/"folds"/("subject_%02d"%subject)
    if (out/"model.pt").exists():
        raise FileExistsError("Preserving existing checkpoint: "+str(out))
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ds=context["dataset_object"]
    config=dict(seed=args.seed,val_size=args.single_val_size if args.protocol=="single_session" else args.val_size,
                test_size=args.test_size,artifact_z=None)
    split=f5._make_split_context(context,subject,config)
    domains=split["domains"]
    source,val,target=(split[k] for k in ("source_ids","val_ids","target_ids"))
    if set(domains[source])&set(domains[target]) or set(domains[source])&set(domains[val]):
        raise ValueError("Independent source/validation/target domains required")
    selected=np.concatenate([source,val,target])
    classes=np.unique(ds["y"][source])
    if not np.array_equal(classes,np.arange(len(classes))):
        raise ValueError("Source labels must be contiguous 0..K-1")
    model=TSMBackboneAblation(variant,ds["x"].shape[1],ds["x"].shape[2],len(classes),
        np.unique(domains[selected]),spatial_filters=args.spatial_filters,subspacedims=args.subspacedims)
    # Materialize only this fold's partitions; no target labels enter refit().
    tensors={}
    for name,ids in (("source",source),("val",val),("target",target)):
        tensors[name]=(torch.from_numpy(split["normalizer"].transform_array(ds["x"][ids])).float(),
                       torch.from_numpy(domains[ids]).long(),torch.from_numpy(ds["y"][ids]).long())
    optimizer=make_optimizer(model,args.lr,args.weight_decay,model_type="tsmnet")
    loss_fn=torch.nn.CrossEntropyLoss()
    best,best_loss,bad,history=None,float("inf"),0,[]
    x,d,y=tensors["source"]
    for epoch in range(1,args.epochs+1):
        model.train()
        total=0.
        for ix in training_batches(d,args.batch_size,args.seed+epoch):
            optimizer.zero_grad()
            loss=loss_fn(model(x[ix],d[ix]),y[ix])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Non-finite training gradient")
            optimizer.step()
            total+=float(loss.detach())*len(ix)
        snapshot=copy.deepcopy(model.state_dict())
        vx,vd,vy=tensors["val"]
        model.eval()
        if model.adaptive:
            model.refit(vx,vd,args.batch_size)
        elif variant=="augspd-spdbn":
            model.refit(x,d,args.batch_size)
        pred=extract(model,vx,vd,args.batch_size)["logits"]
        vloss=float(loss_fn(torch.from_numpy(pred),vy))
        model.load_state_dict(snapshot,strict=True)
        history.append(dict(epoch=epoch,train_loss=total/len(x),val_loss=vloss))
        if vloss<best_loss:
            best_loss,best,bad=vloss,copy.deepcopy(snapshot),0
        else:
            bad+=1
        if bad>=args.patience:
            break
    model.load_state_dict(best,strict=True)
    model.eval()
    if variant=="augspd-spdbn":
        model.refit(x,d,args.batch_size)
    tx,td,ty=tensors["target"]
    if model.adaptive:
        model.refit(tx,td,args.batch_size)
    torch.save(model.state_dict(),out/"model.pt")
    sha=digest(out/"model.pt")
    baseline=extract(model,tx,td,args.batch_size)["logits"]
    snapshot={k:v.detach().clone() for k,v in model.state_dict().items()}
    if model.adaptive:
        model.refit(x,d,args.batch_size)
    changed=[k for k,v in model.state_dict().items() if not torch.equal(v,snapshot[k])]
    allowed={k for k,_ in model.named_buffers(remove_duplicate=False) if any(
        ".batchnorm.dom {}.".format(int(who)) in k or k.startswith("eudsbn.layers.{}.".format(int(who)))
        for who in np.unique(domains[source]))}
    if set(changed)-allowed:
        raise ValueError("Source refit altered weights or non-source buffers: "+str(set(changed)-allowed))
    ids=np.concatenate([source,target])
    values=extract(model,torch.cat([x,tx]),torch.cat([d,td]),args.batch_size)
    actual=values["logits"][len(source):]
    if not np.allclose(actual,baseline,atol=1e-6,rtol=1e-6) or not np.array_equal(actual.argmax(1),baseline.argmax(1)):
        raise ValueError("Source calibration changed target predictions")
    meta=f5._feature_metadata(ds,ids,domains,source)
    meta["fold_id"]=subject
    meta["true_label"]=meta.class_id
    meta.to_csv(out/"samples.csv",index=False)
    archive={k:meta[k].to_numpy(dtype=str if k=="domain" else int) for k in
             ("sample_id","subject_id","fold_id","domain","true_label")}
    np.savez_compressed(out/"paired_spd.npz",**values,**archive)
    audit=dict(passed=True,subject_id=subject,variant=variant,protocol=args.protocol,
        checkpoint_sha256=sha,checkpoint_unchanged=digest(out/"model.pt")==sha,
        target_predictions_identical=True,target_logit_max_abs_delta=float(np.max(np.abs(actual-baseline))),
        changed_source_buffers=changed,source_calibration="source training only",
        target_refit_scope="target_only" if model.adaptive else "none",
        archive_sha256=digest(out/"paired_spd.npz"),metadata_sha256=digest(out/"samples.csv"),
        model_config=dict(variant=variant,nchannels=ds["x"].shape[1],nsamples=ds["x"].shape[2],
            nclasses=len(classes),domains=np.unique(domains[selected]).tolist(),
            spatial_filters=args.spatial_filters,subspacedims=args.subspacedims))
    if "pre" in values:
        audit.update(pre_spd=validate_spd_matrices(values["pre"],"PRE"),
                     post_spd=validate_spd_matrices(values["post"],"POST"))
    f5._write_json(audit,str(out/"audit.json"))
    pd.DataFrame(history).to_csv(out/"history.csv",index=False)
    np.savez_compressed(out/"preprocessing.npz",center=split["normalizer"].center,
        scale=split["normalizer"].scale,source_ids=source,val_ids=val,target_ids=target)
    f5._write_json(dict(arguments=vars(args),split=config,data_cache=context.get("cache"),
        schema="tsmnet_backbone_ablation_v1",best_epoch=int(np.argmin([h["val_loss"] for h in history]))+1),
        str(out/"training_config.json"))
    scores=dict(subject=subject,variant=variant,n_train=len(source),n_val=len(val),n_test=len(target),
        target_adapt=model.adaptive,source_calibrated_accuracy=float((values["logits"][:len(source)].argmax(1)==y.numpy()).mean()),
        test_accuracy=float((actual.argmax(1)==ty.numpy()).mean()),best_val_loss=best_loss,epochs_ran=len(history))
    f5._write_json(scores,str(out/"scores.json"))
    print(scores,flush=True)
    return scores


def load_export(folder):
    audit=json.loads((folder/"audit.json").read_text(encoding="utf-8"))
    if not audit["passed"] or digest(folder/"paired_spd.npz")!=audit["archive_sha256"] or digest(folder/"samples.csv")!=audit["metadata_sha256"]:
        raise ValueError("Export integrity failure")
    z=np.load(folder/"paired_spd.npz",allow_pickle=False)
    meta=pd.read_csv(folder/"samples.csv")
    for key in ("sample_id","subject_id","fold_id","domain","true_label"):
        if not np.array_equal(z[key],meta[key].to_numpy()):
            raise ValueError("Metadata pairing failure")
    return z,meta


def plot_fig5(subjects,args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods=["mean-ce","mean-eudsbn","augspd-spdbn",args.fourth_model]
    plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.family":"DejaVu Sans"})
    fig,axes=plt.subplots(1,6,figsize=(13,3.3))
    rows=[]
    ref_ids={}
    out=Path(args.output_dir)/("fig5_"+args.fourth_model)
    out.mkdir(parents=True,exist_ok=True)
    for j,variant in enumerate(methods):
        for subject in subjects:
            z,meta=load_export(Path(args.output_dir)/variant/"folds"/("subject_%02d"%subject))
            ids=meta.sample_id.to_numpy()
            if subject in ref_ids and not np.array_equal(ref_ids[subject],ids):
                raise ValueError("Methods use different samples")
            ref_ids[subject]=ids
            features=f5._source_standardize(z["features"],meta)
            # Existing helper returns standardized array plus scaler metadata.
            if isinstance(features,tuple):
                features=features[0]
            metrics=f5.high_dimensional_metrics(features,meta)
            rows.append(dict(subject=subject,model=variant,**metrics))
            if subject!=args.representative_subject:
                continue
            manifest=f5.balanced_sample(meta,args.max_points,args.seed)
            positions=f5._manifest_positions(meta,manifest)
            selected=meta.iloc[positions].reset_index(drop=True)
            coords,details=f5.reduce_to_2d(features[positions],selected,reducer=args.reducer,pca_dim=210,seed=args.seed)
            selected["x"],selected["y"]=coords[:,0],coords[:,1]
            selected.to_csv(out/(variant+"_coordinates.csv"),index=False)
            f5._write_json(details,str(out/(variant+"_reducer.json")))
            for ci,c in enumerate(sorted(selected.class_id.unique())):
                for domain,marker in (("source","o"),("target","^")):
                    mask=(selected.class_id==c)&(selected.domain==domain)
                    axes[j].scatter(*coords[mask].T,s=10,alpha=.65,marker=marker,
                                    color=["#0072B2","#D55E00","#009E73"][ci%3])
            axes[j].set_title(variant,fontsize=9)
            axes[j].set_xticks([]); axes[j].set_yticks([])
    frame=pd.DataFrame(rows)
    frame.to_csv(out/"high_dimensional_metrics.csv",index=False)
    for ax,key,title in zip(axes[4:],["domain_discrepancy","separation_ratio"],["Domain discrepancy ↓","Class/domain ratio ↑"]):
        means=frame.groupby("model")[key].mean().reindex(methods)
        ax.plot(range(4),means,"o-")
        if len(subjects)>1:
            spread=frame.groupby("model")[key].std().reindex(methods)
            ax.errorbar(range(4),means,yerr=spread,fmt="none",capsize=2)
        ax.set_xticks(range(4)); ax.set_xticklabels(methods,rotation=55,ha="right",fontsize=6)
        ax.set_title(title,fontsize=8)
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],marker=m,color="gray",ls="",label=d) for d,m in (("Source","o"),("Target","^"))]
    handles += [Line2D([],[],marker="o",color=["#0072B2","#D55E00","#009E73"][i%3],ls="",
                       label=str(selected.loc[selected.class_id==c,"class_name"].iloc[0])) for i,c in enumerate(sorted(selected.class_id.unique()))]
    fig.legend(handles=handles,loc="upper center",ncol=len(handles),frameon=False)
    fig.text(.5,.01,"TSMNet CNN backbone | classifier-input features | quantitative metrics use all high-dimensional samples",ha="center",fontsize=7)
    fig.tight_layout(rect=(0,.06,1,.85))
    for ext in ("pdf","png"):
        fig.savefig(out/("fig5_tsmnet_backbone."+ext),dpi=600)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=["train","plot","all"],default="all")
    p.add_argument("--figures",choices=["all","fig4","fig5"],default="all")
    p.add_argument("--protocol",choices=["single_session","loso"],default="single_session")
    p.add_argument("--datasets",default="stew"); p.add_argument("--dataset-labels",default="STEW")
    p.add_argument("--subjects",default="21"); p.add_argument("--representative-subject",type=int,default=21)
    p.add_argument("--models",default=",".join(VARIANTS))
    p.add_argument("--fourth-model",choices=["tsmnet","augspd-spddsbn"],default="tsmnet")
    p.add_argument("--output-dir",default="outputs/tsmnet_backbone_ablation_v1")
    p.add_argument("--data-root",default="data"); p.add_argument("--cache-root",default="outputs/cache")
    p.add_argument("--epochs",type=int,default=30); p.add_argument("--patience",type=int,default=8)
    p.add_argument("--batch-size",type=int,default=16); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--lr",type=float,default=.001); p.add_argument("--weight-decay",type=float,default=.0001)
    p.add_argument("--val-size",type=float,default=.2); p.add_argument("--single-val-size",type=float,default=.125)
    p.add_argument("--test-size",type=float,default=.2)
    p.add_argument("--spatial-filters",type=int,default=40); p.add_argument("--subspacedims",type=int,default=20)
    p.add_argument("--threads",type=int,default=1); p.add_argument("--max-points",type=int,default=200)
    p.add_argument("--reducer",choices=["tsne","umap"],default="tsne")
    args=p.parse_args()
    for k in ("epochs","patience","batch_size","spatial_filters","subspacedims","threads","max_points"):
        if getattr(args,k)<1: p.error(k+" must be positive")
    torch.set_num_threads(args.threads)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    models=args.models.split(",")
    if not set(models)<=set(VARIANTS): p.error("Unknown model")
    if args.stage=="all" and args.figures in ("all","fig5") and not {"mean-ce","mean-eudsbn","augspd-spdbn",args.fourth_model}<=set(models):
        p.error("Fig5 needs mean-ce,mean-eudsbn,augspd-spdbn and --fourth-model; use --stage train for individual models")
    if args.stage in ("train","all"):
        if (out/"experiment.json").exists(): raise FileExistsError("Use a fresh output directory or --stage plot")
        args.target_fs_stew=args.target_fs_eegmat=args.target_fs_cog_bci=None
        specs=f5._parse_datasets(args.datasets,args.dataset_labels)
        if len(specs)!=1: p.error("One dataset per experiment")
        context=f5._load_dataset_context(specs[0],args)
        subjects=sorted(context["dataset_object"]["meta"].subject.unique().astype(int)) if args.subjects=="all" else list(map(int,args.subjects.split(",")))
        scores=[]
        for subject in subjects:
            for variant in models:
                print("Training",subject,variant,flush=True)
                scores.append(train_fold(context,subject,variant,args))
        pd.DataFrame(scores).to_csv(out/"summary.csv",index=False)
        f5._write_json(dict(subjects=subjects,models=models,arguments=vars(args)),str(out/"experiment.json"))
    if args.stage in ("plot","all"):
        manifest=json.loads((out/"experiment.json").read_text(encoding="utf-8"))
        if manifest["arguments"]["protocol"]!=args.protocol: raise ValueError("Protocol mismatch")
        subjects=manifest["subjects"]
        if args.figures in ("all","fig5") and not {"mean-ce","mean-eudsbn","augspd-spdbn",args.fourth_model}<=set(manifest["models"]):
            raise ValueError("Requested Fig5 models absent from this experiment")
        if args.representative_subject not in subjects: raise ValueError("Representative subject absent")
        if args.figures in ("all","fig5"):
            plot_fig5(subjects,args)
        for variant in (("tsmnet","augspd-spddsbn") if args.figures in ("all","fig4") else ()):
            if variant not in manifest["models"]: continue
            config=argparse.Namespace(output_dir=str(out/variant),representative_subject=args.representative_subject,
                k=15,max_pairs=50000,seed=args.seed,distance_block=256,mean_tolerance=1e-7,mean_iterations=200,plot_points=600)
            rows=[f4.analyze_fold(out/variant/"folds"/("subject_%02d"%s),config) for s in subjects]
            frame=pd.DataFrame(rows); stats=f4.statistics(frame,args.seed)
            frame.to_csv(out/variant/"spddsbn_paired_metrics_per_fold.csv",index=False)
            stats.to_csv(out/variant/"spddsbn_paired_statistics.csv",index=False)
            f4.plot_and_report(frame,stats,config)
            report=out/variant/"SPDDSBN_PAIRED_ANALYSIS_SUMMARY.md"
            content=report.read_text(encoding="utf-8")
            if args.protocol=="single_session":
                content=content.replace("LOSO folds share training subjects; bootstrap intervals and tests do not establish independent replication.",
                    "Evaluation units are within-subject time-block experiments, not LOSO folds; one subject does not support population inference.")
            report.write_text("Model: "+variant+"; protocol: "+args.protocol+"\n\n"+content,encoding="utf-8")


if __name__=="__main__":
    main()
