import pandas as pd
from analysis.fig7_baseline_inventory import inspect_run, inventory


def test_inventory_finds_unregistered_complete_svm_and_detects_missing_checkpoint(tmp_path):
    folder = tmp_path / 'stew_loso_svm'
    folder.mkdir()
    pd.DataFrame(dict(dataset=['stew'] * 48, protocol=['loso'] * 48,
                      model_type=['svm'] * 48, subject=list(range(1, 49)),
                      test_bacc=[.6] * 48)).to_csv(folder / 'summary.csv', index=False)
    for subject in range(1, 49):
        fold = folder / f'subject_{subject:02d}'
        fold.mkdir(); (fold / 'model.joblib').touch()
    rows, _, _ = inventory([], [str(tmp_path)], ['stew'])
    svm = next(r for r in rows if r['method'] == 'SVM')
    assert svm['status'] == 'FILES_PRESENT' and svm['master_rows'] == 0
    (folder / 'subject_48' / 'model.joblib').unlink()
    audit = inspect_run(folder, 'stew', 'svm')
    assert not audit['file_audit_pass'] and audit['missing_checkpoints'] == [48]


def test_inventory_excludes_single_session_and_named_ablation(tmp_path):
    master = tmp_path / 'master.csv'
    pd.DataFrame([
        dict(dataset='stew', protocol='single_session', model_type='svm', model='svm'),
        dict(dataset='stew', protocol='loso', model_type='ms_tgc_spddsbn', model='ms_tgc_spddsbn_chebk1'),
        dict(dataset='stew', protocol='loso', model_type='mdtn-gmda', model='mdtn-gmda', output_dir='missing'),
    ]).to_csv(master, index=False)
    rows, _, _ = inventory([str(master)], [], ['stew'])
    status = {r['method']: r['status'] for r in rows}
    assert status['SVM'] == status['AGMNet (Ours)'] == 'NOT_FOUND'
    assert status['MDTN-GMDA'] == 'MASTER_ONLY'
