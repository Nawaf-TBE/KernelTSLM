"""CNN alignment, masks, class weighting and diagnostic definitions."""
import numpy as np
import pytest
import torch

from kernel_xray.phase3.cnn import CNNConfig, L1TemporalCNN, bounded_class_weights, masked_cross_entropy
from kernel_xray.phase3.cnn_metrics import classification_metrics, confusion_matrix, temporal_masks


def labeled_batch(y):
    y = torch.tensor(y, dtype=torch.int64)
    return {'targets': {'workload_l1': y}, 'masks': {'valid_l1': y >= 0}}


def test_model_shape_parameter_count_and_input_boundary():
    torch.manual_seed(7)
    model = L1TemporalCNN().eval()
    assert model(torch.randn(2, 24, 512)).shape == (2, 12, 512)
    assert model.parameter_count == 126540
    assert model.config.receptive_field == 57
    with pytest.raises(ValueError, match='counter tensor'):
        model({'X': torch.randn(2, 24, 512), 'targets': 'LEAK'})


def test_exact_receptive_field_and_shift_alignment():
    torch.manual_seed(7)
    model = L1TemporalCNN().eval()
    x = torch.randn(1, 24, 512, requires_grad=True)
    model(x)[0, 0, 256].backward()
    active = torch.nonzero(x.grad.abs().sum(1)[0] > 1e-12, as_tuple=False).flatten()
    assert active[0] == 228 and active[-1] == 284
    assert x.grad[:, :, :228].count_nonzero() == x.grad[:, :, 285:].count_nonzero() == 0
    with torch.no_grad():
        shifted = model(x.detach().roll(1, -1))
        original = model(x.detach()).roll(1, -1)
    torch.testing.assert_close(shifted[:, :, 30:480], original[:, :, 30:480], atol=1e-6, rtol=1e-5)


def test_masked_loss_manual_unknown_gradient_and_weight_denominator():
    logits = torch.zeros(1, 12, 512, requires_grad=True)
    y = [[0, 1] + [-1] * 510]
    batch = labeled_batch(y)
    loss = masked_cross_entropy(logits, batch)
    torch.testing.assert_close(loss, torch.tensor(np.log(12), dtype=torch.float32))
    loss.backward()
    assert logits.grad[:, :, 2:].count_nonzero() == 0
    weights = torch.ones(12)
    weights[0], weights[1] = 2, 4
    torch.testing.assert_close(masked_cross_entropy(logits, batch, weights), loss * 3)
    changed = logits.detach().clone()
    changed[:, :, 2:] = 10000 * torch.randn(1, 12, 510)
    torch.testing.assert_close(masked_cross_entropy(changed, batch), loss)


def test_zero_labels_skip_without_weight_decay_step():
    model = L1TemporalCNN()
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1)
    initial = [p.detach().clone() for p in model.parameters()]
    batch = labeled_batch([[-1] * 512])
    loss = masked_cross_entropy(model(torch.randn(1, 24, 512)), batch)
    if loss is not None:
        loss.backward()
        optimizer.step()
    assert loss is None and not optimizer.state
    assert all(torch.equal(before, after) for before, after in zip(initial, model.parameters()))


def test_class_weights_training_expectation_ratio_and_zero_support():
    counts = torch.tensor([1000, 100, 1] + [0] * 9)
    weights = bounded_class_weights(counts, 4)
    positive = weights[counts > 0]
    assert positive.max() / positive.min() <= 4.000001
    assert torch.equal(weights[counts == 0], torch.zeros(9))
    torch.testing.assert_close((weights * counts).sum() / counts.sum(), torch.tensor(1.0))


def test_metrics_fixed_support_zero_division_and_micro():
    cm = confusion_matrix(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    result = classification_metrics(cm, list(range(11)), [str(i) for i in range(12)])
    assert result['accuracy'] == result['micro_f1'] == 0.75
    assert result['per_class'][0]['f1'] == pytest.approx(2/3)
    assert result['per_class'][1]['f1'] == pytest.approx(0.8)
    assert result['macro_f1_supported'] == pytest.approx((2/3 + 0.8) / 11)
    assert result['macro_f1_all12'] == pytest.approx((2/3 + 0.8) / 12)
    assert result['per_class'][11]['f1'] == result['per_class'][11]['support'] == 0


def test_temporal_masks_do_not_bridge_unknowns():
    y = np.array([0] * 12 + [1] * 12 + [-1] * 3 + [2] * 9)
    interior, boundary = temporal_masks(y)
    assert np.flatnonzero(boundary).tolist() == list(range(8, 16))
    assert np.flatnonzero(interior).tolist() == list(range(4, 8)) + list(range(16, 20)) + [31]
    assert not boundary[24:].any()
    assert not (interior & (y < 0)).any()


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_finite_backward(device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS not available')
    torch.manual_seed(7)
    model = L1TemporalCNN().to(device)
    x = torch.randn(2, 24, 512, device=device)
    batch = labeled_batch([[0, 1] + [-1] * 510, [1, 2] + [-1] * 510])
    batch = {'targets': {k: v.to(device) for k, v in batch['targets'].items()},
             'masks': {k: v.to(device) for k, v in batch['masks'].items()}}
    loss = masked_cross_entropy(model(x), batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
