import numpy as np
import pandas as pd
import torch

from src.cl_tsmnet.splits import make_split
from src.cl_tsmnet.training import refit_batchnorm


def _synthetic_loso_dataset(subjects=6, windows_per_subject=8):
    subject = np.repeat(np.arange(1, subjects + 1), windows_per_subject)
    return {
        "name": "synthetic",
        "meta": pd.DataFrame({
            "subject": subject,
            "session": np.ones(len(subject), dtype=np.int64),
        }),
    }


def test_loso_train_validation_and_test_subjects_are_disjoint():
    dataset = _synthetic_loso_dataset()
    split = make_split(
        dataset, "loso", eval_subject=3, seed=42, val_size=0.2
    )
    subject = dataset["meta"]["subject"].to_numpy()
    train_subjects = set(subject[split["train"]])
    val_subjects = set(subject[split["val"]])
    test_subjects = set(subject[split["test"]])

    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)
    assert val_subjects.isdisjoint(test_subjects)
    assert test_subjects == {3}


class _CaptureDomainRefit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.spddsbnorm = torch.nn.Identity()
        self.seen_x = None
        self.seen_domains = None

    def domainadapt_finetune_batched(self, x, d, target_domains,
                                     batch_size=64):
        self.seen_x = x.clone()
        self.seen_domains = d.clone()


def test_domain_refit_receives_unlabelled_target_windows_only():
    model = _CaptureDomainRefit()
    x = np.arange(6, dtype=np.float32).reshape(6, 1, 1)
    domains = np.asarray([1, 1, 2, 2, 3, 3], dtype=np.int64)

    refit_batchnorm(
        model, x, domains,
        train_idx=np.asarray([0, 1, 2, 3]),
        test_idx=np.asarray([4, 5]),
        device=torch.device("cpu"),
        target_adapt=True,
        batch_size=1,
    )

    assert model.seen_x.flatten().tolist() == [4.0, 5.0]
    assert model.seen_domains.tolist() == [3, 3]
