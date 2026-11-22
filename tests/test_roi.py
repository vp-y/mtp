import numpy as np

from bus.roi import crop, keep_largest_component, mask_to_box, postprocess


def test_box_with_margin():
    m = np.zeros((100, 100), bool)
    m[40:60, 30:70] = True  # 20 high, 40 wide
    x0, y0, x1, y1 = mask_to_box(m, margin=0.1, min_size=1)
    assert (x1 - x0, y1 - y0) == (48, 24)
    assert x0 < 30 and x1 > 70 and y0 < 40 and y1 > 60


def test_box_is_clamped_to_image():
    m = np.zeros((50, 50), bool)
    m[0:10, 0:10] = True
    x0, y0, x1, y1 = mask_to_box(m, margin=0.5, min_size=1)
    assert x0 == 0 and y0 == 0 and x1 <= 50 and y1 <= 50


def test_empty_mask_gives_none():
    assert mask_to_box(np.zeros((10, 10), bool)) is None


def test_min_size_and_square():
    m = np.zeros((100, 100), bool)
    m[50, 50] = True
    x0, y0, x1, y1 = mask_to_box(m, margin=0.0, min_size=16, square=True)
    assert x1 - x0 == y1 - y0 == 16


def test_largest_component_drives_box():
    m = np.zeros((100, 100), bool)
    m[10:40, 10:40] = True  # big blob
    m[90:92, 90:92] = True  # small stray blob
    box = mask_to_box(m, margin=0.0, min_size=1, component="largest")
    assert box == (10, 10, 40, 40)
    assert keep_largest_component(m).sum() == 900
    assert postprocess(m, "largest").sum() == 900
    assert postprocess(m, "none", min_area=10).sum() == 900


def test_crop():
    img = np.arange(100).reshape(10, 10)
    assert crop(img, (2, 3, 5, 7)).shape == (4, 3)
