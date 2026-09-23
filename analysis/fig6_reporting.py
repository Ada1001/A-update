"""Figure 6 reporting from real exported subject maps; no model fitting."""
import json
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


def paired_subject_analysis(pre, post, subjects, n_bootstrap=5000, seed=42):
    """Subject-weight bootstrap of a paired, off-diagonal correlation statistic.

    Repeated identities increase pair weights; self-pairs are excluded. This is
    an exploratory conditional interval, NOT a test of independent LOSO repeats.
    """
    from analysis.fig6_node_response_topomaps import compute_topomap_consistency
    if n_bootstrap < 100 or len(subjects) < 4:
        raise ValueError("Need >=4 subjects and >=100 bootstrap replicates")
    pre, post = np.asarray(pre), np.asarray(post)
    if pre.shape != post.shape or len(pre) != len(subjects):
        raise ValueError("PRE/POST subjects and shapes must match")
    a, b = (compute_topomap_consistency(x) for x in (pre, post))
    valid = a['mask'] & b['mask']
    if not valid.any():
        raise ValueError("No common valid subject pairs for consistency analysis")
    pairs = np.asarray(a['pairs'])[valid]
    x, y = a['correlations'][valid], b['correlations'][valid]
    delta = y-x
    rows = []
    for i, subject in enumerate(subjects):
        incident = np.any(pairs == i, axis=1)
        remaining = ~incident
        rows.append(dict(subject_id=int(subject), valid_partners=int(incident.sum()),
                         pre=float(x[incident].mean()) if incident.any() else np.nan,
                         post=float(y[incident].mean()) if incident.any() else np.nan,
                         change=float(delta[incident].mean()) if incident.any() else np.nan,
                         leave_one_out_change=float(delta[remaining].mean()) if remaining.any() else np.nan))
    rng = np.random.default_rng(seed)
    n = len(subjects)
    estimates = []
    for _ in range(n_bootstrap):
        counts = np.bincount(rng.integers(0, n, n), minlength=n)
        weights = counts[pairs[:, 0]] * counts[pairs[:, 1]]
        if weights.sum():
            estimates.append(np.dot(weights, delta)/weights.sum())
    if len(estimates) < .9*n_bootstrap:
        raise ValueError("Too few valid bootstrap draws; inspect constant subject maps")
    lo, hi = np.quantile(estimates, [.025, .975])
    summary = dict(n_subjects=n, valid_pairs=len(delta), consistency_pre=float(x.mean()),
                   consistency_post=float(y.mean()), change=float(delta.mean()),
                   conditional_bootstrap_low=float(lo), conditional_bootstrap_high=float(hi),
                   requested_replicates=n_bootstrap, valid_replicates=len(estimates), seed=seed,
                   pairs_increased=int((delta>0).sum()),
                   subjects_increased=sum(r['change']>0 for r in rows),
                   loo_min=float(np.nanmin([r['leave_one_out_change'] for r in rows])),
                   loo_max=float(np.nanmax([r['leave_one_out_change'] for r in rows])),
                   inference='Exploratory subject-weight bootstrap conditional on fitted models; '
                             'excludes self-pairs; overlapping LOSO training not modeled; no p value.')
    return summary, pd.DataFrame(rows), np.asarray(estimates)


def plot_ab_three_rows(datasets, args):
    """Three task rows, true class means, vertical explicitly scoped scales."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from analysis.fig6_node_response_topomaps import draw_topomap
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':6.5,'pdf.fonttype':42,'ps.fonttype':42})
    prepared=[]
    for d in datasets:
        ids=[d['subjects'].index(r['subject_id']) for r in d['selection']]
        a=[]; b=[]
        for suffix in ['', '_high', '_low']:
            pre,post=d['pre'+suffix],d['post'+suffix]
            if not np.isfinite(pre).all() or not np.isfinite(post).all(): raise ValueError('Non-finite task maps')
            a.append(np.stack([pre.mean(0),post.mean(0)]))
            b.append(np.concatenate([post[ids],post.mean(0)[None]],axis=0))
        if not np.allclose(a[0],a[1]-a[2]) or not np.allclose(b[0],b[1]-b[2]):
            raise ValueError('High-Low does not match the exported class means')
        prepared.append((d,a,b))
    n=len(datasets)
    fig=plt.figure(figsize=(9.2,4.55*n+.35))
    # Dedicated spacer columns protect vertical colorbar ticks from map titles.
    grid=fig.add_gridspec(3*n,12,width_ratios=[1,1,.10,.68,1,1,1,1,1,.10,.58,.01],
                         left=.105,right=.98,bottom=.075/n,top=.90,hspace=.42,wspace=.12)
    limits=[]; statistics=[]; notes=[]
    for k,(d,a,b) in enumerate(prepared):
        pool=prepared if args.color_scope=='global' else [(d,a,b)]
        alimit=max(float(abs(pa[0]).max()) for _,pa,_ in pool)
        blimit=max(float(abs(pb[0]).max()) for _,_,pb in pool)
        arange=(min(float(np.min(pa[1:])) for _,pa,_ in pool),max(float(np.max(pa[1:])) for _,pa,_ in pool))
        brange=(min(float(np.min(pb[1:])) for _,_,pb in pool),max(float(np.max(pb[1:])) for _,_,pb in pool))
        if args.color_mode=='shared-original':
            alimit=blimit=max(float(np.abs(np.r_[dd['pre'],dd['post']]).max()) for dd,_,_ in pool)
            arange=brange=(min(float(np.min(np.r_[dd['pre_high'],dd['pre_low'],dd['post_high'],dd['post_low']])) for dd,_,_ in pool),
                           max(float(np.max(np.r_[dd['pre_high'],dd['pre_low'],dd['post_high'],dd['post_low']])) for dd,_,_ in pool))
        alimit=max(alimit,1e-12); blimit=max(blimit,1e-12)
        arange=(arange[0],max(arange[1],arange[0]+1e-12)); brange=(brange[0],max(brange[1],brange[0]+1e-12))
        d['color_limit']=float(np.abs(np.r_[d['pre'],d['post']]).max())
        for r,label in enumerate(['High - Low','High workload','Low workload']):
            row=3*k+r
            av=(-alimit,alimit) if r==0 else arange
            bv=(-blimit,blimit) if r==0 else brange
            cmap='RdBu_r'
            maps=[]
            for j,values in enumerate(list(a[r])+list(b[r])):
                col=j if j<2 else j+2
                ax=fig.add_subplot(grid[row,col]); maps.append(ax)
                v=av if j<2 else bv
                draw_topomap(ax,values,d['xy'],v[1],args.backend,vmin=v[0],cmap=cmap)
                if r==0:
                    title=['Pre-graph','Post-graph']+["S%02d"%x['subject_id'] for x in d['selection']]+['Avg']
                    ax.set_title(title[j],fontsize=7,pad=3)
            pos=maps[0].get_position()
            fig.text(.009,pos.y0+pos.height/2,label,va='center',fontsize=6.5,weight='bold')
            if r==0:
                fig.text(pos.x0,pos.y1+.058/n,d['display_name']+'  |  (A) Group means',fontsize=8,weight='bold')
                fig.text(maps[2].get_position().x0,pos.y1+.058/n,'(B) Post-graph subject examples',fontsize=8,weight='bold')
            for block,col,v in [('A',2,av),('B',9,bv)]:
                cax=fig.add_subplot(grid[row,col])
                cb=fig.colorbar(ScalarMappable(norm=Normalize(*v),cmap=cmap),cax=cax,orientation='vertical')
                cb.ax.tick_params(labelsize=5.5,pad=1,length=2)
                cb.set_label('Contrast (a.u.)' if r==0 else 'Response (a.u.)',fontsize=5.5,labelpad=2)
                limits.append(dict(dataset=d['display_name'],row=label,panel=block,vmin=v[0],vmax=v[1],
                                   scope='group means' if block=='A' else 'displayed subjects + all-subject Avg',
                                   high_low_shared=True,clipped_electrode_values=0))
        summary,_,_=paired_subject_analysis(d['pre'],d['post'],d['subjects'],getattr(args,'bootstrap_replicates',5000),args.seed)
        summary['dataset']=d['display_name']; statistics.append(summary)
        notes.append('{}: group contrast ±{:.6g}; displayed individual contrast ±{:.6g}; High/Low group range {}; High/Low individual range {}.'.format(d['display_name'],alimit,blimit,arange,brange))
    fig.text(.5,.015/n,'A and B use separately labeled scales. High and Low share ranges within each panel. Colors do not indicate statistical significance.',ha='center',fontsize=6)
    for ext in ['pdf','png']: fig.savefig(out/('Fig6_node_response_topomaps.'+ext),dpi=600)
    plt.close(fig)
    pd.DataFrame(limits).to_csv(out/'fig6_color_limits.csv',index=False)
    pd.DataFrame(statistics).to_csv(out/'fig6_exploratory_statistics.csv',index=False)
    (out/'fig6_reporting_notes.md').write_text('# Fig6 A/B three-row layout\n\n'+'\n\n'.join(notes)+
        '\n\nOnly A/B are drawn: High-Low, High, Low. A shows equal-subject PRE/POST means; B shows unchanged BAcc-selected subjects and the all-subject POST mean. '
        'Sign consistency is omitted because class-specific L2 responses are nonnegative and their signs are not an informative counterpart. '
        'All rows use RdBu_r. Contrast scales are symmetric around zero. Class means use shared High/Low ranges, which need not begin at zero; blue/red indicate lower/higher positive response, not negative/positive values. '
        'Default B limits use the displayed subjects and Avg, explicitly changing the previous all-subject scale. No electrode values are clipped or maps normalized. '
        'Absolute colors across A/B must be read against their own colorbars. Interpolation may overshoot measured electrode extrema. '
        'C/D are omitted; exploratory conditional statistics remain in CSV, not significance labels. They do not account for overlapping LOSO training sets.\n',encoding='utf-8')
    return statistics


def plot_all_subjects(datasets, args):
    """Additional POST High-Low atlas with a fixed shared display range."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from analysis.fig6_node_response_topomaps import draw_topomap
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42})
    limits=[]
    for d in datasets:
        post=np.asarray(d['post'])
        subjects=d['subjects']
        if len(subjects)!=len(post) or not np.isfinite(post).all():
            raise ValueError('Invalid all-subject POST maps')
        limit=float(getattr(args,'all_subjects_limit',10.))
        if not np.isfinite(limit) or limit<=0: raise ValueError('Atlas limit must be finite and positive')
        below=int((post < -limit).sum()); above=int((post > limit).sum())
        ncols=min(8,len(subjects)); nrows=int(np.ceil(len(subjects)/ncols))
        fig=plt.figure(figsize=(1.35*ncols+1.1,1.48*nrows+.6))
        grid=fig.add_gridspec(nrows,ncols,left=.025,right=.895,bottom=.05,top=.94,
                             wspace=.12,hspace=.22)
        for slot,i in enumerate(np.argsort(subjects)):
            # Display order is subject ID, not accuracy or map appearance.
            ax=fig.add_subplot(grid[slot//ncols,slot%ncols])
            draw_topomap(ax,post[i],d['xy'],limit,args.backend)
            ax.set_title('S%02d'%subjects[i],fontsize=9,pad=3)
        cax=fig.add_axes([.915,.20,.015,.60])
        cb=fig.colorbar(ScalarMappable(norm=Normalize(-limit,limit),cmap='RdBu_r'),
                        cax=cax,orientation='vertical',extend='both',ticks=np.linspace(-limit,limit,5))
        cb.set_label('Post-graph High - Low response (a.u.)',fontsize=9)
        fig.suptitle(d['display_name']+' | All exported subjects | Post-graph High - Low',fontsize=12,y=.985)
        fig.text(.46,.015,'Shared scale: [-{:g}, {:g}]; out-of-range colors saturate; original values unchanged.'.format(limit,limit),ha='center',fontsize=8)
        tag=''.join(c if c.isalnum() or c in '-_' else '_' for c in d['display_name']).lower()
        for ext in ['pdf','png']:
            fig.savefig(out/('Fig6_all_subjects_'+tag+'_post_contrast.'+ext),dpi=600)
        plt.close(fig)
        limits.append(dict(dataset=d['display_name'],n_subjects=len(subjects),vmin=-limit,vmax=limit,
                           scope='all exported subject POST contrasts',normalization='none',
                           below_limit=below,above_limit=above,total_electrode_values=int(post.size),
                           original_min=float(post.min()),original_max=float(post.max()),
                           display_policy='saturate colors only; no numerical clipping'))
    pd.DataFrame(limits).to_csv(out/'fig6_all_subjects_color_limits.csv',index=False)


def plot_report(datasets, args):
    plot_all_subjects(datasets,args)
    if getattr(args,"layout","diagnostic")=="ab-three-rows":
        return plot_ab_three_rows(datasets,args)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from analysis import fig6_node_response_topomaps as f
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':6.5,
                         'pdf.fonttype':42, 'ps.fonttype':42})
    count = len(datasets)
    fig = plt.figure(figsize=(7.16, 3.65*count+.3))
    grid = fig.add_gridspec(2*count, 9, width_ratios=[1,1,1,.3,1,1,1,1,1],
                          height_ratios=[1.4,1]*count, left=.065, right=.98,
                          top=.90, bottom=.15/count, hspace=1.15, wspace=.35)
    mode = getattr(args, 'color_mode', 'group-detail')
    global_raw = max(np.abs(np.r_[d['pre'],d['post']]).max() for d in datasets)
    global_group = max(np.abs(np.stack([d['pre'].mean(0),d['post'].mean(0)])).max() for d in datasets)
    global_individual = max(np.abs(d['post']).max() for d in datasets)
    stats, changes, limits, notes = [], [], [], []
    for k,d in enumerate(datasets):
        pre, post = d['pre'], d['post']
        shared = global_raw if args.color_scope=='global' else np.abs(np.r_[pre,post]).max()
        group = global_group if args.color_scope=='global' else np.abs(np.stack([pre.mean(0),post.mean(0)])).max()
        individual = global_individual if args.color_scope=='global' else np.abs(post).max()
        if mode=='shared-original': group=individual=shared
        group, individual = max(float(group),1e-12), max(float(individual),1e-12)
        d['color_limit'] = float(shared)  # preserved raw audit limit
        d['group_color_limit'], d['individual_color_limit'] = group, individual
        limits.append(dict(dataset=d['display_name'],mode=mode,group_limit=group,
                           individual_limit=individual,raw_all_subjects_limit=float(shared)))
        axes = [fig.add_subplot(grid[2*k,j]) for j in (0,1,2,4,5,6,7,8)]
        f.plot_group_level_topomaps(axes[:3],pre,post,d['xy'],group,args.backend)
        f.plot_representative_subjects(axes[3:],post,d['subjects'],d['selection'],d['xy'],individual,args.backend)
        for ax in axes: ax.set_title(ax.get_title(), fontsize=6.5, pad=3)
        first, third, fourth, last = [axes[j].get_position() for j in (0,2,3,7)]
        fig.text(first.x0,first.y1+.055/count,d['display_name']+'  |  (A) Group maps',weight='bold',fontsize=7)
        fig.text(fourth.x0,first.y1+.055/count,'(B) BAcc-selected examples',weight='bold',fontsize=7)
        for start,end,limit,label,ticks in [
            (first,axes[1].get_position(),group,'Group High - Low (a.u.)',None),
            (third,third,1.,'Mean sign',[-1,0,1]),
            (fourth,last,individual,'Individual / Avg High - Low (a.u.)',None)]:
            cax=fig.add_axes([start.x0,first.y0-.042/count,end.x1-start.x0,.012/count])
            cb=fig.colorbar(ScalarMappable(norm=Normalize(-limit,limit),cmap='RdBu_r'),
                            cax=cax,orientation='horizontal',ticks=ticks)
            cb.ax.tick_params(labelsize=5.5,pad=1)
            cb.set_label(label,fontsize=5.5,labelpad=2)
        summary, rows, draws = paired_subject_analysis(pre,post,d['subjects'],
                                    getattr(args,'bootstrap_replicates',5000),getattr(args,'seed',42))
        summary['dataset']=d['display_name']; stats.append(summary)
        rows.insert(0,'dataset',d['display_name']); changes.append(rows)
        pd.DataFrame({'change':draws}).to_csv(out/(d['display_name'].lower()+'_conditional_bootstrap.csv'),index=False)
        ax=fig.add_subplot(grid[2*k+1,:4])
        for r in rows.itertuples(): ax.plot([0,1],[r.pre,r.post],color='#8294a3',alpha=.45,lw=.55)
        ax.plot([0,1],[summary['consistency_pre'],summary['consistency_post']],'-o',color='#005a8d',lw=1.5,ms=3,label='Pairwise mean')
        ax.set(xticks=[0,1],xticklabels=['Pre','Post'],ylabel='Mean r to other subjects',ylim=(-1,1),xlim=(-.3,1.3))
        ax.set_title('(C) Subject-wise spatial agreement',loc='left',fontsize=7)
        ax=fig.add_subplot(grid[2*k+1,5:])
        ax.axvline(0,color='.6',lw=.7,ls='--')
        ax.plot([summary['conditional_bootstrap_low'],summary['conditional_bootstrap_high']],[1,1],color='#005a8d',lw=2)
        ax.scatter([summary['change']],[1],color='#005a8d',s=16,zorder=3)
        ax.plot([summary['loo_min'],summary['loo_max']],[0,0],color='#9c572e',lw=2)
        ax.set(yticks=[0,1],yticklabels=['Delete-one range','Conditional 95% interval'],ylim=(-.6,1.7),xlabel='Change in mean r (Post - Pre)')
        ax.set_title('(D) Exploratory sensitivity',loc='left',fontsize=7)
        for ax in fig.axes:
            if ax in axes: continue
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        notes.append('{}: PRE {:.6f}, POST {:.6f}, change {:.6f}; conditional bootstrap interval [{:.6f}, {:.6f}]; delete-one range [{:.6f}, {:.6f}].'.format(
            d['display_name'],summary['consistency_pre'],summary['consistency_post'],summary['change'],
            summary['conditional_bootstrap_low'],summary['conditional_bootstrap_high'],summary['loo_min'],summary['loo_max']))
    fig.text(.5,.025/count,'Group / individual color scales are separately labeled. Intervals condition on fitted models; no significance test.',
             ha='center',fontsize=5.5)
    for ext in ['pdf','png']: fig.savefig(out/('Fig6_node_response_topomaps.'+ext),dpi=600)
    plt.close(fig)
    pd.DataFrame(stats).to_csv(out/'fig6_exploratory_statistics.csv',index=False)
    pd.concat(changes,ignore_index=True).to_csv(out/'fig6_subject_consistency_changes.csv',index=False)
    pd.DataFrame(limits).to_csv(out/'fig6_color_limits.csv',index=False)
    (out/'fig6_reporting_notes.md').write_text('# Fig6 reporting v2\n\n'+'\n\n'.join(notes)+
        '\n\nPRE/POST group means share a symmetric scale. Individual maps and Avg share a separate scale based on ALL subject POST maps; no selected-subject scale fitting. '
        'Shared-original mode uses one raw response scale. No map-wise normalization or clipping of electrode values. Sign agreement uses [-1,1].\n\n'
        'Paired subject-weight bootstrap uses 5000 draws by default, weighting each distinct pair by sampled subject multiplicities and excluding identity self-pairs. '
        'Both phases use the same weights and common valid pairs. Percentile intervals are exploratory and conditional on these fitted models. '
        'They do not account for overlapping LOSO training sets, model retraining uncertainty, or dataset/task confounds. '
        'An interval excluding zero is not presented as a confirmatory significance result. No p values or stars are generated. '
        'Delete-one ranges are sensitivity ranges, not confidence intervals. Subject-wise lines share partners and are dependent.\n',encoding='utf-8')
    return stats


def replot_export(source, args):
    """Re-render audited exports without EEG caches, GPU, or checkpoints."""
    root=Path(source).resolve(); out=Path(args.output_dir).resolve()
    if out==root: raise ValueError('Use a different output directory to preserve original exports')
    contrast=pd.read_csv(root/'fig6_subject_contrasts.csv')
    selected=pd.read_csv(root/'fig6_subject_selection.csv')
    rows=[]
    for name,frame in contrast.groupby('dataset',sort=False):
        layout=pd.read_csv(root/(name+'_layout.csv')); channels=layout.channel.astype(str).tolist()
        if len(set(channels))!=len(channels): raise ValueError('Duplicate layout channels')
        subjects=sorted(frame.subject_id.unique().astype(int).tolist())
        arrays={phase:frame[frame.phase==phase].pivot(index='subject_id',columns='channel',values='contrast').loc[subjects,channels].to_numpy() for phase in ['pre','post']}
        class_maps={phase+"_"+label:frame[frame.phase==phase].pivot(index="subject_id",columns="channel",values=label+"_response").loc[subjects,channels].to_numpy()
                    for phase in ["pre","post"] for label in ["high","low"]}
        for i,s in enumerate(subjects):
            folder=root/'node_response_cache'/name/('subject_%02d'%s)
            audit=json.loads((folder/'audit.json').read_text(encoding='utf-8'))
            archive=folder/'responses.npz'
            if hashlib.sha256(archive.read_bytes()).hexdigest()!=audit['response_sha256'] or not audit['passed'] or not audit['state_unchanged']:
                raise ValueError('Export audit failed: '+str(folder))
            with np.load(archive,allow_pickle=False) as z:
                labels=np.unique(z['label'])
                if len(labels)!=2: raise ValueError('Binary labels required')
                # Recover contrast orientation from exported class means, allowing custom high_label.
                order=[list(z['channels'].astype(str)).index(c) for c in channels]
                signs=[]
                for phase in ['pre','post']:
                    delta=(z[phase][z['label']==labels[-1]].mean(0)-z[phase][z['label']==labels[0]].mean(0))[order]
                    signs.append([np.allclose(sign*delta,arrays[phase][i],rtol=1e-9,atol=1e-9) for sign in [1,-1]])
                for phase in ['pre','post']:
                    means=[z[phase][z['label']==label].mean(0)[order] for label in labels]
                    valid_orientation=[]
                    for lo,hi in [(0,1),(1,0)]:
                        valid_orientation.append(np.allclose(means[hi],class_maps[phase+'_high'][i]) and
                                                 np.allclose(means[lo],class_maps[phase+'_low'][i]))
                    if not any(valid_orientation): raise ValueError('Class means export mismatch: '+str(folder))
                if not np.any(np.all(signs,axis=0)): raise ValueError('Contrast export mismatch: '+str(folder))
        selection=selected[selected.dataset==name].to_dict('records')
        if len(selection)!=4 or len({r['subject_id'] for r in selection})!=4: raise ValueError('Need four distinct exported representatives')
        rows.append(dict(display_name=name.upper(),pre=arrays['pre'],post=arrays['post'],xy=layout[['x','y']].to_numpy(),subjects=subjects,selection=selection,**class_maps))
    result=plot_report(rows,args)
    files=[root/'fig6_subject_contrasts.csv',root/'fig6_subject_selection.csv']
    files.extend(root/(name+'_layout.csv') for name in contrast.dataset.unique())
    files.extend(sorted((root/'node_response_cache').glob('*/*/audit.json')))
    (out/'fig6_replot_provenance.json').write_text(json.dumps(dict(source=str(root),arguments=vars(args),
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2),encoding='utf-8')
    (out/'fig6_summary.md').write_text(
        '# Fig6 replot from audited real exports\n\nOriginal export: '+str(root)+'\n\n'
        'No training, EEG preprocessing, model inference, subject reselection, or response normalization was performed. '
        'Response hashes and both contrast phases were checked against exported response arrays.\n\n'+
        (out/'fig6_reporting_notes.md').read_text(encoding='utf-8'),encoding='utf-8')
    print(pd.DataFrame(result).to_string(index=False))
