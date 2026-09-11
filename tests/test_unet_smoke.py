import torch

from eoflood.models import build_model


def test_unet_scratch_forward():
    cfg = {"model": {"name": "unet", "tag": "unet_scratch", "encoder": "resnet18", "encoder_weights": None}}
    model = build_model(cfg, in_channels=6)
    x = torch.randn(2, 6, 64, 64)
    out = model(x)
    assert out.shape == (2, 2, 64, 64)


def test_train_step_decreases_loss():
    torch.manual_seed(0)
    cfg = {"model": {"name": "unet", "tag": "unet_scratch", "encoder": "resnet18", "encoder_weights": None}}
    model = build_model(cfg, in_channels=6)
    x = torch.randn(4, 6, 32, 32)
    y = (x[:, 0] > 0).long()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    losses = []
    for _ in range(15):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_resolve_epochs_keeps_step_budget():
    from eoflood.train import resolve_epochs

    assert resolve_epochs({"epochs": 50}, 31) == 50
    assert resolve_epochs({"epochs": 50, "total_steps": 1550}, 31) == 50
    assert resolve_epochs({"epochs": 50, "total_steps": 1550}, 3) == 517   # 10 % labels
    assert resolve_epochs({"epochs": 50, "total_steps": 1550}, 1) == 1550  # 5 % labels


def test_partial_checkpoint_roundtrip(tmp_path):
    from eoflood.train import load_checkpoint, trainable_state_dict

    cfg = {"model": {"name": "unet", "tag": "unet_scratch", "encoder": "resnet18", "encoder_weights": None}}
    model = build_model(cfg, in_channels=6)
    # freeze the encoder: the checkpoint must then contain only decoder/head params + buffers
    for p in model.encoder.parameters():
        p.requires_grad = False
    sd = trainable_state_dict(model)
    assert not any(k.startswith("encoder.") and not k.endswith(("running_mean", "running_var", "num_batches_tracked")) for k in sd)
    torch.save({"model": sd, "epoch": 0, "cfg": cfg}, tmp_path / "best.pt")
    fresh = build_model(cfg, in_channels=6)
    for p in fresh.encoder.parameters():
        p.requires_grad = False
    load_checkpoint(fresh, tmp_path / "best.pt", torch.device("cpu"))
    # the loaded (trainable) tensors must match exactly; the frozen encoder is rebuilt, not loaded
    for k, v in sd.items():
        assert torch.equal(fresh.state_dict()[k], v)
