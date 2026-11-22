"""Model construction and forward passes, without downloading pretrained weights."""
import pytest
import torch

from bus.models.segmentation import VanillaUNet, build_seg_model


def test_vanilla_unet_shape():
    out = VanillaUNet(1, base=8)(torch.zeros(2, 1, 64, 64))
    assert tuple(out.shape) == (2, 1, 64, 64)


@pytest.mark.parametrize("arch", ["unet", "fpn", "deeplabv3plus"])
@pytest.mark.parametrize("encoder", ["resnet34", "efficientnet-b0"])
def test_smp_models(arch, encoder):
    pytest.importorskip("segmentation_models_pytorch")
    m = build_seg_model({"arch": arch, "encoder": encoder, "encoder_weights": None, "base_channels": 32}, 1)
    m.eval()
    with torch.no_grad():
        out = m(torch.zeros(1, 1, 256, 256))
    assert tuple(out.shape) == (1, 1, 256, 256)


@pytest.mark.parametrize("backbone", ["mobilenetv3_small", "efficientnet_b0", "resnet18"])
@pytest.mark.parametrize("head", ["linear", "mlp"])
def test_classifier(backbone, head):
    pytest.importorskip("timm")
    from bus.models.classification import Classifier, set_trainable

    m = Classifier(backbone, head, 0.5, pretrained=False, img_size=224)
    n_frozen = set_trainable(m, 0)
    n_k10 = set_trainable(m, 10)
    n_all = set_trainable(m, "all")
    assert n_frozen < n_k10 < n_all
    m.train()
    out = m(torch.zeros(2, 3, 224, 224))
    assert tuple(out.shape) == (2, 1)


def test_multitask_swin():
    pytest.importorskip("timm")
    from bus.baselines.multitask_swin import MultiTaskSwin

    m = MultiTaskSwin(pretrained=False).eval()
    with torch.no_grad():
        seg, cls = m(torch.zeros(1, 3, 224, 224))
    assert tuple(seg.shape) == (1, 1, 224, 224) and tuple(cls.shape) == (1, 2)


def test_maskrcnn_builds():
    pytest.importorskip("torchvision")
    from bus.baselines.maskrcnn import build_maskrcnn

    m = build_maskrcnn(256, 256, pretrained=False).eval()
    with torch.no_grad():
        out = m([torch.rand(3, 256, 256)])[0]
    assert {"boxes", "labels", "scores", "masks"} <= set(out)
