"""Real-checkpoint paired SPDDSBN analysis. No training or fabricated observations."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from analysis import fig5_representation_alignment as f5
from analysis import diagnose_fig5_alignment as diag
from src.cl_tsmnet.spd_pca import (airm_karcher_mean, common_tangent_vectors,
    validate_spd_matrices, migrate_legacy_spddsbn_buffers)
from src.cl_tsmnet.spd_visualization_adapters import extract_spd_intermediates


def distances(a, b):
    """Broadcasted exact AIRM, float64, without clipping invalid eigenvalues."""
    s, u = np.linalg.eigh(a)
    if np.min(s) <= 0:
        raise ValueError("AIRM received non-SPD input")
    inv = (u * (1 / np.sqrt(s))[..., None, :]) @ u.swapaxes(-1, -2)
    w = inv @ b @ inv
    e = np.linalg.eigvalsh((w + w.swapaxes(-1, -2)) / 2)
    if not np.isfinite(e).all() or np.min(e) <= 0:
        raise ValueError("AIRM whitening lost positive definiteness")
    return np.sqrt(np.square(np.log(e)).sum(-1))


def fisher(x, y):
    center = x.mean(0)
    between = within = 0.0
    for c in np.unique(y):
        group = x[y == c]
        mean = group.mean(0)
        between += len(group) * np.square(mean - center).sum()
        within += np.square(group - mean).sum()
    if within <= 0 or len(np.unique(y)) < 2:
        raise ValueError("Fisher ratio undefined")
    return float(between / (within + 1e-12))


def structure(pre, post, k=15, max_pairs=50000, seed=42, block=256):
    """Exact pooled-domain kNN with O(n + block*d*d) working memory.

    Spearman uses uniformly sampled distinct unordered pairs, same before/after.
    Ties in neighbors use stable original sample order.
    """
    n = len(pre)
    if n <= k:
        raise ValueError("Domain sample count must exceed k")
    total = n * (n - 1) // 2
    ranks = np.sort(np.random.default_rng(seed).choice(total, min(total, max_pairs), replace=False))
    ends = np.cumsum(np.arange(n - 1, 0, -1, dtype=np.int64))
    rows = np.searchsorted(ends, ranks, side="right")
    starts = np.r_[0, ends[:-1]]
    cols = rows + 1 + ranks - starts[rows]
    pdist = np.empty((2, len(ranks)))
    overlaps = []
    for i in range(n):
        neighbors = []
        for condition, matrices in enumerate((pre, post)):
            d = np.empty(n)
            for start in range(0, n, block):
                d[start:start + block] = distances(matrices[i], matrices[start:start + block])
            selected = rows == i
            pdist[condition, selected] = d[cols[selected]]
            d[i] = np.inf
            neighbors.append(np.argsort(d, kind="stable")[:k])
        overlaps.append(len(np.intersect1d(*neighbors)) / k)
        if i % 500 == 0:
            print("  exact AIRM neighbors {}/{}".format(i, n), flush=True)
    rho = float(spearmanr(*pdist).statistic)
    if not np.isfinite(rho):
        raise ValueError("Spearman undefined (constant distances)")
    return float(np.mean(overlaps)), rho, len(ranks)


def export_fold(context, info, method, subject, args):
    out = Path(args.output_dir) / "folds" / ("subject_%02d" % subject)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "audit.json").exists():
        raise FileExistsError("Completed export exists; use --stage analyze or a new output directory")
    dataset, domains = context["dataset_object"], context["domains"]
    split = f5._make_split_context(context, subject, f5._split_config(info))
    source, target = split["source_ids"], split["target_ids"]
    if np.intersect1d(domains[source], domains[target]).size:
        raise ValueError("Source and target domains overlap")
    ids = np.concatenate([source, target])
    selected = np.concatenate([source, split["val_ids"], target])
    checkpoint = f5._checkpoint_path(info["run_dir"], subject)
    sha = diag.digest(checkpoint)
    model = f5._build_model(dataset, domains, selected, source, method,
                           f5._model_config(info["record"]), torch.device(args.device))
    state, migration = migrate_legacy_spddsbn_buffers(f5._load_state(checkpoint), model.state_dict())
    model.load_state_dict(state, strict=True)
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    baseline = diag.extract(model, dataset, domains, target, split["normalizer"],
                            method["model_type"], args.batch_size, torch.device(args.device))["logits"]
    diag.refit_source(model, dataset, domains, source, split["normalizer"],
                      method["model_type"], args.batch_size, torch.device(args.device))
    changed = diag.audit_state(before, model, np.unique(domains[source]))
    chunks = {"pre": [], "post": [], "logits": []}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ids), args.batch_size):
            ix = ids[start:start + args.batch_size]
            x = torch.from_numpy(split["normalizer"].transform_array(dataset["x"][ix])).to(args.device)
            d = torch.from_numpy(domains[ix]).long().to(args.device)
            logits, intermediate = extract_spd_intermediates(model, x, d, method["model_type"])
            for name, value in (("pre", intermediate["spd_pre_bn"]),
                                ("post", intermediate["spd_post_bn"]), ("logits", logits)):
                chunks[name].append(value.detach().cpu().numpy())
    values = {k: np.concatenate(v) for k, v in chunks.items()}
    actual = values["logits"][len(source):]
    audit = {"subject_id": subject, "checkpoint": str(checkpoint), "checkpoint_sha256": sha,
             "checkpoint_unchanged": diag.digest(checkpoint) == sha,
             "target_logits_allclose": bool(np.allclose(baseline, actual, rtol=1e-6, atol=1e-6)),
             "target_predictions_identical": bool(np.array_equal(baseline.argmax(1), actual.argmax(1))),
             "target_logit_max_abs_delta": float(np.max(np.abs(baseline - actual))),
             "changed_source_buffers": changed, "migrations": migration,
             "pre_location": "BiMap + ReEig output immediately before SPDDSBN",
             "post_location": "SPDDSBN output immediately before LogEig",
             "calibration": "all source training windows only; saved target statistics untouched",
             "run_record": info["record"], "split_config": split["config"],
             "pre_spd": validate_spd_matrices(values["pre"], "PRE"),
             "post_spd": validate_spd_matrices(values["post"], "POST")}
    if not all(audit[k] for k in ("checkpoint_unchanged", "target_logits_allclose", "target_predictions_identical")):
        raise RuntimeError("Calibration invariant failed: " + str(audit))
    meta = f5._feature_metadata(dataset, ids, domains, source)
    meta["fold_id"] = subject
    meta["true_label"] = meta.class_id
    meta.to_csv(out / "samples.csv", index=False)
    np.savez_compressed(out / "paired_spd.npz", **values, sample_id=ids,
                        subject_id=meta.subject_id.to_numpy(), fold_id=np.full(len(ids), subject),
                        domain=meta.domain.to_numpy(dtype=str), true_label=meta.class_id.to_numpy())
    audit["archive_sha256"] = diag.digest(out / "paired_spd.npz")
    audit["metadata_sha256"] = diag.digest(out / "samples.csv")
    audit["passed"] = True
    f5._write_json(audit, str(out / "audit.json"))


def analyze_fold(folder, args):
    audit = json.loads((folder / "audit.json").read_text(encoding="utf-8"))
    if not audit["passed"] or diag.digest(folder / "paired_spd.npz") != audit["archive_sha256"] or diag.digest(folder / "samples.csv") != audit["metadata_sha256"]:
        raise ValueError("Missing or invalid export audit: " + str(folder))
    z = np.load(folder / "paired_spd.npz", allow_pickle=False)
    meta = pd.read_csv(folder / "samples.csv")
    for name in ("sample_id", "subject_id", "fold_id", "domain", "true_label"):
        if not np.array_equal(z[name], meta[name].to_numpy()):
            raise ValueError("Paired metadata mismatch: " + name)
    if meta.sample_id.duplicated().any() or z["pre"].shape != z["post"].shape:
        raise ValueError("Invalid pairing")
    pre, post = z["pre"], z["post"]
    validate_spd_matrices(pre, "PRE")
    validate_spd_matrices(post, "POST")
    signature = {"version": 1, "archive": audit["archive_sha256"],
                 "k": args.k, "max_pairs": args.max_pairs, "seed": args.seed,
                 "mean_tolerance": args.mean_tolerance, "mean_iterations": args.mean_iterations}
    cache = folder / "analysis_complete.json"
    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved["signature"] == signature and all(diag.digest(folder / name) == sha for name,sha in saved["files"].items()):
            print("  Reusing audited completed metrics", flush=True)
            return json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    source = meta.domain.to_numpy() == "source"
    y = meta.true_label.to_numpy()
    def mean(x):
        return airm_karcher_mean(x, tolerance=args.mean_tolerance,
                                 max_iterations=args.mean_iterations)[0]
    reference = mean(pre[source])
    a, b = (common_tangent_vectors(x, reference) for x in (pre, post))
    scaler = StandardScaler().fit(a[source])
    a, b = scaler.transform(a), scaler.transform(b)
    pca = PCA(n_components=2, svd_solver="full").fit(a[source])
    coords = {"pre": pca.transform(a), "post": pca.transform(b)}
    np.savez_compressed(folder / "common_coordinates.npz", **coords, reference=reference,
                        scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                        pca_mean=pca.mean_, pca_components=pca.components_,
                        pca_explained_variance_ratio=pca.explained_variance_ratio_)
    row = {"fold_id": int(audit["subject_id"]), "subject_id": int(audit["subject_id"]),
           "n_source": int(source.sum()), "n_target": int((~source).sum())}
    for name, matrices, vectors in (("pre", pre, a), ("post", post, b)):
        class_distances = []
        for c in np.unique(y):
            sm, tm = source & (y == c), ~source & (y == c)
            if not sm.any() or not tm.any():
                raise ValueError("Class absent from source/target")
            class_distances.append(float(distances(mean(matrices[sm]), mean(matrices[tm]))))
        row["domain_discrepancy_" + name] = float(np.mean(class_distances))
        for partition, mask in (("source", source), ("target", ~source), ("pooled", np.ones(len(y), bool))):
            row["fisher_" + partition + "_" + name] = fisher(vectors[mask], y[mask])
        row["fisher_" + name] = row["fisher_target_" + name]
    for partition, mask in (("source", source), ("target", ~source)):
        print(" structure:", partition, flush=True)
        knn, rho, pairs = structure(pre[mask], post[mask], args.k, args.max_pairs, args.seed, args.distance_block)
        row["knn_" + partition], row["rho_" + partition] = knn, rho
        row["pairs_" + partition] = pairs
        row["knn_preservation_" + partition] = knn
        row["distance_rank_" + partition] = rho
    f5._write_json(row, str(folder / "metrics.json"))
    f5._write_json({"signature": signature, "files": {name: diag.digest(folder / name) for name in
                    ("metrics.json", "common_coordinates.npz")}}, str(cache))
    return row


def statistics(frame, seed):
    rows = []
    for key in frame.columns:
        if not key.startswith(("domain_discrepancy", "fisher", "knn", "rho")):
            continue
        x = frame[key].to_numpy()
        rows.append(dict(metric=key, n=len(x), mean=x.mean(), std=x.std(ddof=1),
                         median=np.median(x), q25=np.quantile(x, .25), q75=np.quantile(x, .75),
                         iqr=np.quantile(x, .75)-np.quantile(x, .25)))
    for prefix in ("domain_discrepancy", "fisher_source", "fisher_target", "fisher_pooled"):
        delta = (frame[prefix + "_post"] - frame[prefix + "_pre"]).to_numpy()
        nonzero = delta[delta != 0]
        ranks = rankdata(np.abs(nonzero))
        effect = float(np.sum(ranks * np.sign(nonzero)) / ranks.sum()) if len(nonzero) else 0.
        boot = np.random.default_rng(seed).choice(delta, (10000, len(delta)), replace=True)
        lo, hi = np.quantile(np.median(boot, axis=1), [.025, .975])
        rows.append(dict(metric=prefix + "_delta_post_minus_pre", n=len(delta), mean=delta.mean(),
                         std=delta.std(ddof=1), median=np.median(delta), q25=np.quantile(delta, .25),
                         q75=np.quantile(delta, .75), iqr=np.quantile(delta,.75)-np.quantile(delta,.25),
                         p_wilcoxon=float(wilcoxon(delta).pvalue) if len(nonzero) else 1.,
                         rank_biserial=effect, median_ci_low=lo, median_ci_high=hi))
    return pd.DataFrame(rows)


def plot_and_report(frame, stats, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    out = Path(args.output_dir)
    subject = args.representative_subject
    folder = out / "folds" / ("subject_%02d" % subject)
    meta = pd.read_csv(folder / "samples.csv")
    xy = np.load(folder / "common_coordinates.npz")
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    classes = sorted(meta.true_label.unique())
    if len(classes) > len(colors):
        raise ValueError("Too many classes for configured palette")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "axes.titlesize": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.8))
    rng = np.random.default_rng(args.seed)
    # Display sampling is domain-balanced and label-blind; metrics use ALL samples.
    shown = []
    for domain in ("source", "target"):
        available = np.flatnonzero(meta.domain.to_numpy() == domain)
        shown.extend(rng.choice(available, min(len(available), args.plot_points // 2), replace=False))
    shown = np.sort(shown)
    limits = np.concatenate([xy["pre"], xy["post"]])
    lo, hi = limits.min(0), limits.max(0)
    pad = np.maximum((hi-lo)*.06, 1e-5)
    for panel, key in enumerate(("pre", "post")):
        ax = axes[0, panel]
        for ci, c in enumerate(classes):
            for domain, marker in (("source", "o"), ("target", "^")):
                ix = shown[(meta.true_label.to_numpy()[shown] == c) & (meta.domain.to_numpy()[shown] == domain)]
                ax.scatter(*xy[key][ix].T, s=9 if marker == "o" else 18, c=colors[ci], marker=marker,
                           alpha=.4 if marker == "o" else .8, linewidths=.25,
                           edgecolors="none" if marker == "o" else "black")
        ax.set_title("({}) {} SPDDSBN".format("ab"[panel], "Before" if key == "pre" else "After"))
    ax = axes[0, 2]
    for ci, c in enumerate(classes):
        for domain, marker in (("source", "o"), ("target", "^")):
            mask = (meta.true_label.to_numpy() == c) & (meta.domain.to_numpy() == domain)
            start, end = (xy[key][mask].mean(0) for key in ("pre", "post"))
            ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color=colors[ci], lw=1))
            ax.scatter(*start, marker=marker, s=25, facecolors="none", edgecolors=colors[ci])
            ax.scatter(*end, marker=marker, s=25, c=colors[ci])
    ax.set_title("(c) Projected centroid shifts")
    for ax in axes[0]:
        ax.set(xlim=(lo[0]-pad[0],hi[0]+pad[0]), ylim=(lo[1]-pad[1],hi[1]+pad[1]), xlabel="Common PC1", ylabel="Common PC2")
    def paired(ax, prefix, title):
        x = frame[[prefix + "_pre", prefix + "_post"]].to_numpy()
        for row in x:
            ax.plot([0, 1], row, "o-", ms=2, lw=.4, color=".65", alpha=.6)
        for i in range(2):
            boot = np.random.default_rng(args.seed).choice(x[:, i], (10000, len(x)), replace=True)
            lower, upper = np.quantile(np.median(boot, axis=1), [.025, .975])
            mid = np.median(x[:, i])
            ax.errorbar(i, mid, yerr=[[mid-lower], [upper-mid]], fmt="o", color="black", capsize=3, ms=4)
        ax.set(xticks=[0, 1], xticklabels=["Before", "After"], title=title, xlim=(-.3,1.3))
    paired(axes[1, 0], "domain_discrepancy", "(d) Class-conditional AIRM ↓")
    ax = axes[1, 1]
    for i, key in enumerate(("knn_source", "knn_target", "rho_source", "rho_target")):
        values = frame[key].to_numpy()
        ax.boxplot(values, positions=[i], widths=.45, showfliers=False, medianprops=dict(color="black"))
        ax.scatter(i + rng.uniform(-.12,.12,len(values)), values, s=7, color=".4", marker="o" if key.endswith("source") else "^")
    ax.set(xticks=range(4), xticklabels=["kNN\nS", "kNN\nT", "ρ\nS", "ρ\nT"], ylim=(-1.05,1.05), title="(e) Within-domain structure")
    paired(axes[1, 2], "fisher_target", "(f) Target Fisher ratio ↑")
    handles = [Line2D([], [], marker="o", ls="", color=colors[i], label=str(meta.loc[meta.true_label == c,"class_name"].iloc[0])) for i,c in enumerate(classes)]
    handles += [Line2D([], [], marker=m, ls="", color=".4", label=d) for d,m in (("Source","o"),("Target","^"))]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(.5,.018,"Subject {}: common PRE-source PCA; (c) open → filled. All {} folds: median and bootstrap 95% CI.".format(subject,len(frame)),ha="center",fontsize=6)
    fig.tight_layout(rect=(0,.05,1,.93), w_pad=1.2,h_pad=1.6)
    for ext in ("pdf", "png"):
        fig.savefig(out / ("Fig_SPDDSBN_Paired_Mechanism." + ext), dpi=600)
    plt.close(fig)
    lines = ["# SPDDSBN paired mechanism analysis", "", "Real checkpoint exports; all export/SPD/pairing audits passed.",
             "PRE is the BiMap + ReEig output; POST is immediately after SPDDSBN. This isolates the layer inside a trained model, not an untrained/raw-data baseline.",
             "All exported source-training and target-test windows enter metrics. Source kNN pools source subjects. Target labels are used only for post-hoc class metrics and colors.",
             "AIRM reference, standardization and PCA are fitted only to PRE source windows. Fisher uses all standardized tangent dimensions; panel (f) uses target Fisher.",
             "P values are exploratory, two-sided, unadjusted Wilcoxon tests. LOSO folds share training subjects; bootstrap intervals and tests do not establish independent replication.", ""]
    selected = frame.loc[frame.subject_id == subject].iloc[0]
    for key in ("domain_discrepancy", "fisher_target"):
        delta = frame[key+"_post"] - frame[key+"_pre"]
        st = stats.loc[stats.metric == key+"_delta_post_minus_pre"].iloc[0]
        lines.append("{}: mean PRE {:.6g}, POST {:.6g}; decreases in {}/{} folds; median paired change {:.6g}, 95% bootstrap CI [{:.6g}, {:.6g}], p={:.6g}, rank-biserial={:.4g}.".format(key,frame[key+"_pre"].mean(),frame[key+"_post"].mean(),int((delta<0).sum()),len(frame),st["median"],st.median_ci_low,st.median_ci_high,st.p_wilcoxon,st.rank_biserial))
    for key in ("domain_discrepancy_post", "fisher_target_post", "knn_source", "knn_target", "rho_source", "rho_target"):
        x = frame[key]
        percentile = 100 * ((x < selected[key]).sum() + .5*(x == selected[key]).sum()) / len(x)
        near_median = int(frame.loc[(x-x.median()).abs().idxmin(),"subject_id"])
        near_mean = int(frame.loc[(x-x.mean()).abs().idxmin(),"subject_id"])
        lines.append("{}: mean {:.5g}, median {:.5g}; subject {} percentile {:.1f}%; nearest median fold {}, nearest mean fold {}.{}".format(key,x.mean(),x.median(),subject,percentile,near_median,near_mean," WARNING: representative fold is extreme on this metric." if percentile<10 or percentile>90 else ""))
    dstat = stats.loc[stats.metric == "domain_discrepancy_delta_post_minus_pre"].iloc[0]
    jstat = stats.loc[stats.metric == "fisher_target_delta_post_minus_pre"].iloc[0]
    lines.append("Domain discrepancy: " + ("the median paired change is negative, with its descriptive interval below zero." if dstat.median_ci_high < 0 else "the interval does not support a clear median reduction."))
    lines.append("Target discrimination: " + ("increased (median-change interval above zero)." if jstat.median_ci_low > 0 else "decreased (median-change interval below zero)." if jstat.median_ci_high < 0 else "direction uncertain; maintenance is not established."))
    lines.append("Relational structure: report the measured kNN overlap as the retained fraction of 15 neighbors, and rho as distance-order consistency, separately for pooled source and target. No validated cutoff is available here for a binary well-preserved claim.")
    lines += ["", "Interpretation: assess domain discrepancy, kNN overlap, distance rank correlation and Fisher jointly. There is no predeclared equivalence margin for structure or discrimination, so these results alone do not establish preservation/equivalence. A nonsignificant Fisher change is not evidence of maintenance.",
              "This figure is suitable for reporting the observed paired mechanism diagnostics; whether it supports the intended scientific claim depends on the actual directions, magnitudes and uncertainty above. Do not select another subject or reducer to obtain a preferred conclusion."]
    (out / "SPDDSBN_PAIRED_ANALYSIS_SUMMARY.md").write_text("\n\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["all", "extract", "analyze"], default="all")
    p.add_argument("--datasets", default="stew")
    p.add_argument("--dataset-labels", default="STEW")
    p.add_argument("--subjects", default="all")
    p.add_argument("--expected-folds", type=int, default=48)
    p.add_argument("--representative-subject", type=int, default=21)
    p.add_argument("--output-root", default="outputs/fig5_stew_v3")
    p.add_argument("--master-summary", default="outputs/fig5_stew_v3/master_summary.csv")
    p.add_argument("--data-root", default="data")
    p.add_argument("--cache-root", default="outputs/cache")
    p.add_argument("--output-dir", default="results/fig4_spddsbn")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--max-pairs", type=int, default=50000)
    p.add_argument("--distance-block", type=int, default=256)
    p.add_argument("--plot-points", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mean-tolerance", type=float, default=1e-7)
    p.add_argument("--mean-iterations", type=int, default=200)
    args = p.parse_args()
    for name in ("batch_size", "k", "max_pairs", "distance_block", "plot_points", "expected_folds", "mean_iterations"):
        if getattr(args,name) <= 0:
            p.error(name + " must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    args.target_fs_stew = args.target_fs_eegmat = args.target_fs_cog_bci = None
    args.allow_missing_master_config = args.allow_legacy_refit = False
    if args.stage in ("all", "extract"):
        specs = f5._parse_datasets(args.datasets,args.dataset_labels)
        if len(specs) != 1:
            p.error("Use one dataset per directory")
        method = f5._load_methods(None)[-1]
        if method["model_type"] != "ms_tgc_spddsbn":
            raise ValueError("Expected full model")
        info = f5._resolve_run(specs[0],method,args,pd.read_csv(args.master_summary))
        f5._validate_comparable_runs([info],[method],args)
        subjects = sorted(set(info["summary"].subject.astype(int))) if args.subjects == "all" else sorted(set(map(int,args.subjects.split(","))))
        if len(subjects) != args.expected_folds:
            raise ValueError("Fold count differs from --expected-folds")
        context = f5._load_dataset_context(specs[0],args)
        for subject in subjects:
            print("Exporting subject",subject,flush=True)
            export_fold(context,info,method,subject,args)
        f5._write_json({"subjects": subjects, "arguments": vars(args)},str(out / "export_manifest.json"))
    if args.stage in ("all", "analyze"):
        manifest = json.loads((out / "export_manifest.json").read_text(encoding="utf-8"))
        subjects = manifest["subjects"]
        if len(subjects) != args.expected_folds or args.representative_subject not in subjects:
            raise ValueError("Unexpected fold count or representative subject absent")
        rows = []
        for subject in subjects:
            print("Analyzing subject",subject,flush=True)
            rows.append(analyze_fold(out / "folds" / ("subject_%02d" % subject),args))
        frame = pd.DataFrame(rows)
        stats = statistics(frame,args.seed)
        frame.to_csv(out / "spddsbn_paired_metrics_per_fold.csv",index=False)
        stats.to_csv(out / "spddsbn_paired_statistics.csv",index=False)
        f5._write_json(vars(args),str(out / "analysis_arguments.json"))
        plot_and_report(frame,stats,args)


if __name__ == "__main__":
    main()
