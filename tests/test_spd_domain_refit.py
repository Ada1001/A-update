import copy
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TSMNET_ROOT = PROJECT_ROOT / "TSMNet"
if str(TSMNET_ROOT) not in sys.path:
    sys.path.insert(0, str(TSMNET_ROOT))

import spdnets.batchnorm as bn


def test_unseen_domain_refit_preserves_buffer_shapes_and_state_restore():
    layer = bn.AdaMomDomainSPDBatchNorm(
        (1, 4, 4),
        batchdim=0,
        domains=torch.tensor([1, 2]),
        learn_mean=False,
        learn_std=True,
        dispersion=bn.BatchNormDispersion.SCALAR,
        dtype=torch.double,
        device=torch.device("cpu"),
    )
    snapshot = copy.deepcopy(layer.state_dict())
    shapes_before = {key: value.shape for key, value in snapshot.items()}

    layer.eval()
    layer.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)
    x = torch.eye(4, dtype=torch.double).reshape(1, 1, 4, 4).repeat(3, 1, 1, 1)
    x = x + 0.05 * torch.randn_like(x)
    x = torch.matmul(x, x.transpose(-1, -2)) + 0.1 * torch.eye(4, dtype=torch.double)
    layer(x, torch.full((3,), 2, dtype=torch.long))

    shapes_after = {
        key: value.shape for key, value in layer.state_dict().items()
    }
    assert shapes_after == shapes_before
    layer.load_state_dict(snapshot)
