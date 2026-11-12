"""The THINGS-pretrained pixel network (data/things_pixel_network/best.pt): a spatial adapter per subject, one temporal
layer shared by all channels, one hidden layer, 32x32 RGB output. Input (n, 31, 250) in STARSTIM_31 order, ZCA x1.1 as the
THINGS lab-contract cache; lab data go through the adapter's last slot, which no THINGS subject used and is the identity.

SubjectAdapter and PixelReadout are copied from atms_pipeline/subject_adapter.py of Hierarchical_EEG2Image_Reconstruction
(branch dev/lab_lowlevel_transfer, commits e9c83fa / 807bcf0); the checkpoint was trained there by
scripts/train_things_lowlevel_pretrain.py on the THINGS-EEG2 lab-contract data of 10 subjects.
"""
import functools

import torch
from torch import nn

from . import data_things
from .paths import THINGS_PIXEL_NETWORK


class SubjectAdapter(nn.Module):
    """Identity-initialised per-subject spatial adapter. x: (B, C, T), subject_ids: (B,) long."""

    def __init__(self, n_subjects: int, n_channels: int, rank: int = 4):
        super().__init__()
        slots = n_subjects + 1
        self.gain = nn.Parameter(torch.ones(slots, n_channels))
        self.offset = nn.Parameter(torch.zeros(slots, n_channels))
        self.U = nn.Parameter(torch.zeros(slots, n_channels, rank))
        self.V = nn.Parameter(torch.randn(slots, n_channels, rank) * 0.01)

    @property
    def identity_slot(self) -> int:
        return self.gain.shape[0] - 1

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        gain = self.gain[subject_ids].unsqueeze(-1)
        offset = self.offset[subject_ids].unsqueeze(-1)
        projected = torch.einsum("bct,bcr->brt", x, self.V[subject_ids])
        mixed = torch.einsum("bcr,brt->bct", self.U[subject_ids], projected)
        return gain * x + offset + mixed


class PixelReadout(nn.Module):
    """Subject-adapted low-level readout for 32x32 RGB targets."""

    def __init__(self, n_subjects: int, n_channels: int = 31, rank: int = 4):
        super().__init__()
        self.adapter = SubjectAdapter(n_subjects, n_channels, rank)
        self.temporal = nn.Linear(250, 128)
        self.norm = nn.LayerNorm(n_channels * 128)
        self.hidden = nn.Linear(n_channels * 128, 4096)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(4096, 3 * 32 * 32)

    def _readout(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x).flatten(1)
        x = self.norm(x)
        x = self.hidden(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.output(x))
        return x.view(x.shape[0], 3, 32, 32)

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        return self._readout(self.adapter(x, subject_ids))


@functools.lru_cache(maxsize=1)
def payload() -> dict:
    """The checkpoint (format lowlevel_v1): state_dict and config, which lists the THINGS conditions it was trained on."""
    assert THINGS_PIXEL_NETWORK.exists(), (f"{THINGS_PIXEL_NETWORK} missing: set things_pixel_network in "
                                           "configs/paths.local.yaml and run scripts/link_data.py")
    p = torch.load(THINGS_PIXEL_NETWORK, map_location="cpu", weights_only=True)
    assert p.get("format") == "lowlevel_v1" and p.get("arch") == "pixel_readout", (p.get("format"), p.get("arch"))
    return p


def load(device="cuda") -> PixelReadout:
    """A fresh copy of the pretrained network, in eval mode."""
    cfg = payload()["config"]
    model = PixelReadout(n_subjects=int(cfg["n_subjects"]), rank=int(cfg.get("rank", 4)))
    model.load_state_dict(payload()["state_dict"], strict=True)
    return model.to(device).eval()


def pretraining_images() -> tuple[set, set]:
    """File names of the THINGS training images the network was trained on, and of those it held out for its own
    validation (they chose the checkpoint). Every other image it never saw."""
    cfg = payload()["config"]; names = [p.name for p in data_things.image_paths("train")]
    out, val = set(cfg["excluded_condition_indices"]), set(cfg["inner_val_condition_indices"])
    return {nm for i, nm in enumerate(names) if i not in out and i not in val}, {names[i] for i in val}
