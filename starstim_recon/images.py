"""Image targets. pixel32 is the one target definition used for every dataset (THINGS, old lab, sessions)."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

_CROP512 = transforms.Compose([transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
                               transforms.CenterCrop(512), transforms.ToTensor()])


def pixel32(path) -> torch.Tensor:
    """(3, 32, 32) RGB in [0, 1]: bicubic resize to 512, centre crop, 16x16 average pooling."""
    return F.adaptive_avg_pool2d(_CROP512(Image.open(path).convert("RGB")).unsqueeze(0), 32)[0]


def pixel32_many(paths) -> torch.Tensor:
    return torch.stack([pixel32(p) for p in paths])


def to8(x32) -> np.ndarray:
    """(n, 3, 32, 32) -> (n, 8, 8, 3), the coarse colour layout that coarse8 scores."""
    return F.adaptive_avg_pool2d(torch.as_tensor(x32).float(), 8).permute(0, 2, 3, 1).numpy()


def photo(path, res=96) -> np.ndarray:
    img = Image.open(path).convert("RGB"); w, h = img.size; s = min(w, h)
    return np.asarray(img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((res, res)))


def enhanced(pred8, true8, groups) -> np.ndarray:
    """Display only: set mean of the targets + (prediction - set mean of the predictions) x k, one k for all images,
    matching the spread of the targets (ridge predictions are shrunk toward the mean)."""
    dp, dt, base = np.zeros_like(pred8), np.zeros_like(true8), np.zeros_like(true8)
    for g in np.unique(groups):
        m = groups == g
        dp[m] = pred8[m] - pred8[m].mean(0); dt[m] = true8[m] - true8[m].mean(0); base[m] = true8[m].mean(0)
    return np.clip(base + dp * (dt.std() / (dp.std() + 1e-12)), 0, 1)


# ------------------------------------------------------------------------------------------------ global properties
PROPS = ["L", "a", "b", "contrast", "chroma", "sf_slope", "lr_diff", "tb_diff"]
_CROP128 = transforms.Compose([transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
                               transforms.CenterCrop(512), transforms.Resize(128), transforms.ToTensor()])


def lab(rgb):
    """(3, H, W) sRGB in [0, 1] -> CIELAB L*, a*, b* (D65)."""
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = np.einsum("ij,jhw->ihw", M, c) / np.array([0.95047, 1.0, 1.08883])[:, None, None]
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return 116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])


def sf_slope(gray):
    """Slope of log power vs log spatial frequency, 2-40 cycles per image."""
    p = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean()))) ** 2
    n = gray.shape[0]; yy, xx = np.indices(p.shape); r = np.hypot(yy - n // 2, xx - n // 2).astype(int)
    f = np.arange(2, 41); pw = np.array([p[r == k].mean() for k in f])
    return float(np.polyfit(np.log(f), np.log(pw), 1)[0])


def properties(rgb) -> np.ndarray:
    """8 global properties (PROPS) of a (3, H, W) image in [0, 1]."""
    L, a, b = lab(np.clip(rgb, 0, 1)); h, w = L.shape
    return np.array([L.mean(), a.mean(), b.mean(), L.std(), np.hypot(a, b).mean(), sf_slope(L),
                     L[:, : w // 2].mean() - L[:, w // 2:].mean(), L[: h // 2].mean() - L[h // 2:].mean()])


def image_properties(paths) -> np.ndarray:
    return np.stack([properties(_CROP128(Image.open(p).convert("RGB")).numpy().astype(np.float64)) for p in paths])
