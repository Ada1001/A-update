"""Read-only inventory of ten LOSO baselines; does not load or train models."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = {
    'svm': 'SVM', 'eegnet': 'EEGNet', 'bilstm': 'BiLSTM', 'tahag': 'TAHAG',
    'lsccn': 'LSCCN', 'eegconformer': 'EEG-Conformer', 'bfgcn': 'BF-GCN',
    'mdtn': 'MDTN-GMDA', 'tsmnet': 'TSMNet', 'ms_tgc_spddsbn': 'AGMNet (Ours)',
}
ALIASES = {k: {k} for k in METHODS}
ALIASES['mdtn'] = {'mdtn', 'mdtn_gmda', 'mdtn-gmda'}
ALIASES['tsmnet'] = {'tsmnet', 'tsmnet_spddsbn'}


def canonical(value):
    return next((k for k, names in ALIASES.items() if str(value) in names), None)


def inspect_run(folder, dataset, kind):
    result = dict(folder=str(folder), summary_exists=False, file_audit_pass=False)
    p = folder / 'summary.csv'
    if not p.is_file():
        return result
    try:
        df = pd.read_csv(p)
        required = {'dataset', 'protocol', 'model_type', 'subject', 'test_bacc'}
        if not required <= set(df) or df.empty:
            raise ValueError('Missing required fold-summary columns or empty summary')
        if set(df.dataset) != {dataset} or set(df.protocol) != {'loso'} or set(df.model_type.map(canonical)) != {kind}:
            raise ValueError('Fold-summary identity mismatch')
        ids = pd.to_numeric(df.subject, errors='raise').to_numpy(dtype=float)
        if not np.isfinite(ids).all() or not np.equal(ids, ids.astype(int)).all():
            raise ValueError('Invalid subject IDs')
        subjects = ids.astype(int).tolist()
        expected = list(range(1, 49)) if dataset == 'stew' else list(range(36))
        bacc = pd.to_numeric(df.test_bacc, errors='raise')
        valid = bool(np.isfinite(bacc).all() and bacc.between(0, 1).all())
        checkpoint = 'model.joblib' if kind == 'svm' else 'model.pt'
        missing_cp = [s for s in subjects if not (folder / f'subject_{s:02d}' / checkpoint).is_file()]
        missing_history = [s for s in subjects if not any(
            (folder / f'subject_{s:02d}' / n).is_file() for n in ('history.csv', 'epoch_metrics.csv'))]
        result.update(summary_exists=True, n_folds=len(subjects),
                      expected_subjects_match=sorted(subjects) == expected,
                      mean_target_bacc=float(bacc.mean()) if valid else None,
                      missing_checkpoints=missing_cp, missing_histories=missing_history,
                      file_audit_pass=bool(sorted(subjects) == expected and valid and not missing_cp))
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        result['error'] = str(exc)
    return result


def inventory(masters, roots, datasets):
    rows = []
    notes = []
    for filename in masters:
        p = Path(filename)
        if not p.is_file():
            notes.append(f'Master not found: {p}')
            continue
        frame = pd.read_csv(p)
        for r in frame.to_dict('records'):
            kind = canonical(r.get('model_type'))
            if (r.get('protocol') != 'loso' or r.get('dataset') not in datasets
                    or kind is None or str(r.get('model')) not in ALIASES[kind]):
                continue
            rows.append(dict(dataset=r['dataset'], kind=kind, origin=str(p),
                             output_dir=str(r.get('output_dir', '')), n=r.get('n'),
                             bacc=r.get('balanced_accuracy_mean'), timestamp=r.get('timestamp')))
    # Locate completed runs even if master_summary.csv never received their row.
    found = {}
    for root in roots:
        for p in sorted(Path(root).rglob('summary.csv')):
            try:
                h = pd.read_csv(p, nrows=1)
                if h.empty or not {'dataset', 'protocol', 'model_type'} <= set(h):
                    continue
                r = h.iloc[0]; kind = canonical(r.model_type)
                if kind and r.protocol == 'loso' and r.dataset in datasets:
                    # Avoid treating named ablations as the full method.
                    if 'model' in h and str(r['model']) not in ALIASES[kind]:
                        continue
                    if kind == 'ms_tgc_spddsbn' and p.parent.name != f'{r.dataset}_loso_ms_tgc_spddsbn':
                        continue
                    found.setdefault((r.dataset, kind), set()).add(p.parent.resolve())
            except (ValueError, OSError, pd.errors.ParserError) as exc:
                notes.append(f'Cannot inspect {p}: {exc}')
    details = []; summary = []
    for dataset in datasets:
        for kind, method in METHODS.items():
            records = [r for r in rows if r['dataset'] == dataset and r['kind'] == kind]
            folders = set(found.get((dataset, kind), set()))
            for r in records:
                original = Path(r['output_dir'].replace('\\', '/'))
                if original.is_dir():
                    folders.add(original.resolve())
            audits = [inspect_run(p, dataset, kind) for p in sorted(folders)]
            status = ('FILES_PRESENT' if any(a['file_audit_pass'] for a in audits)
                      else 'FILES_INCOMPLETE' if audits
                      else 'MASTER_ONLY' if records else 'NOT_FOUND')
            details.append(dict(dataset=dataset, method=method, records=records, file_checks=audits))
            summary.append(dict(dataset=dataset, method=method, status=status,
                                master_rows=len(records), folders_found=len(folders),
                                complete_file_sets=sum(a['file_audit_pass'] for a in audits)))
    return summary, details, notes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--master-summary', default='outputs/master_summary.csv,outputs/fig7_retrained/master_summary.csv')
    p.add_argument('--roots', default='outputs,output', help='Comma-separated server folders to scan')
    p.add_argument('--datasets', default='stew,eegmat')
    p.add_argument('--output-dir', default='results/fig7_baseline_inventory')
    args = p.parse_args()
    datasets = args.datasets.split(',')
    if not set(datasets) <= {'stew', 'eegmat'}:
        p.error('Expected subject IDs are defined only for stew/eegmat')
    rows, details, notes = inventory(args.master_summary.split(','), args.roots.split(','), datasets)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'baseline_inventory.csv', index=False)
    (out / 'baseline_inventory_details.json').write_text(json.dumps(
        dict(notes=notes, runs=details), indent=2, default=str), encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))
    print('\nFILES_PRESENT checks IDs, metrics and checkpoint file presence only; it does not certify loadability, architecture, or timing compatibility.')
    print('Legacy mdtn-gmda is inventoried separately by path; do not assume it matches the current MDTN implementation.')
    print('SVM has no neural-network epoch curve; missing SVM history is not a failure.')
    for note in notes:
        print(note)


if __name__ == '__main__':
    main()
