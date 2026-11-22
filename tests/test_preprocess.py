import cv2
import numpy as np
import pytest

from bus.data.preprocess import resize_to_square, restore, to_tensor


@pytest.mark.parametrize("mode", ["pad", "stretch"])
def test_mask_roundtrip(mode):
    m = np.zeros((300, 500), np.uint8)
    m[100:200, 150:350] = 1
    r, g = resize_to_square(m, 256, mode, cv2.INTER_NEAREST)
    assert r.shape == (256, 256)
    back = restore(r, g, cv2.INTER_NEAREST)
    assert back.shape == m.shape
    assert (back == m).mean() > 0.99


def test_pad_keeps_aspect_ratio():
    img = np.full((100, 200), 255, np.uint8)
    r, g = resize_to_square(img, 256, "pad")
    ys, xs = np.nonzero(r)
    assert xs.max() - xs.min() + 1 == 256
    assert abs((ys.max() - ys.min() + 1) - 128) <= 1
    assert g.top > 0 and g.left == 0


def test_restore_float_map():
    prob = np.random.rand(256, 256).astype(np.float32)
    _, g = resize_to_square(np.zeros((480, 640), np.uint8), 256, "pad")
    assert restore(prob, g).shape == (480, 640)


def test_to_tensor():
    img = np.random.randint(0, 255, (64, 48), np.uint8)
    assert tuple(to_tensor(img, 1).shape) == (1, 64, 48)
    t3 = to_tensor(img, 3)
    assert tuple(t3.shape) == (3, 64, 48) and str(t3.dtype) == "torch.float32"
    raw = to_tensor(img, 3, normalize=False)
    assert float(raw.min()) >= 0 and float(raw.max()) <= 1
