import pytest
import torch


def test_multiscale_decoder_shapes():
    from eoflood.models.decoders import MultiScaleUNetDecoder

    dec = MultiScaleUNetDecoder(in_channels=32, num_classes=2, width=32, patch=16)
    maps = [torch.randn(2, 32, 14, 14) for _ in range(4)]
    out = dec(maps)
    assert out.shape == (2, 2, 224, 224)
    with pytest.raises(ValueError):
        dec(maps[:3])


@pytest.mark.skipif(not pytest.importorskip("terratorch", reason="terratorch not installed"), reason="")
def test_prithvi_unet_decoder_forward():
    from eoflood.models.prithvi import PrithviSegmenter
    from eoflood.utils import count_params

    m = PrithviSegmenter(finetune="lora", pretrained=False, decoder="unet")
    tr, tot = count_params(m)
    assert 2_000_000 < tr < 20_000_000
    with torch.no_grad():
        assert m(torch.randn(1, 6, 224, 224)).shape == (1, 2, 224, 224)
        assert m(torch.randn(1, 6, 512, 512)).shape == (1, 2, 512, 512)
