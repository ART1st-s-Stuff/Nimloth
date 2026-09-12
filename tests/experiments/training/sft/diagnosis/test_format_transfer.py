import torch

from experiments.training.sft.diagnosis.format_transfer import (
    fit_precision,
    full_vocab_row_loss,
)


def test_full_vocabulary_competitor_changes_loss():
    hidden = torch.tensor([[1., 0.]])
    rows = torch.tensor([[0., 0.]], requires_grad=True)
    ids = torch.tensor([1])
    targets = torch.tensor([1])
    ordinary = full_vocab_row_loss(hidden, rows, ids, targets, torch.zeros(1, 3))
    rival = full_vocab_row_loss(hidden, rows, ids, targets, torch.tensor([[10., 0., 0.]]))
    assert rival > ordinary + 8
    rival.backward()
    assert rows.grad[0, 0] < 0


def test_precision_fit_does_not_change_original_weights():
    hidden = torch.tensor([[1., 2.], [-1., 1.]])
    weights = torch.full((4, 2), .02, dtype=torch.bfloat16)
    before = weights.clone()
    result = fit_precision(hidden, weights, torch.tensor([2, 3]), torch.tensor([2, 3]), steps=2, lr=5e-6)
    assert torch.equal(weights, before)
    assert len(result['torch.bfloat16']['history']) == 2
    assert result['torch.bfloat16']['history'][0]['actual_abs_mean'] == 0
    assert result['torch.float32']['history'][0]['actual_abs_mean'] > 0


def test_probability_comparison_ignores_common_shift_in_kl():
    from experiments.training.sft.diagnosis.format_transfer import (
        probability_comparison,
    )
    logits = torch.tensor([[1., 2., 3.], [-1., 0., 1.]])
    result = probability_comparison(logits, logits + 8)
    assert result['raw_max_abs'] == 8
    assert result['centered_max_abs'] < 1e-6
    assert max(abs(x) for x in result['kl_per_position']) < 1e-6
    assert result['argmax_flip_positions'] == []


def test_token_difference_retains_exact_inputs():
    from experiments.training.sft.diagnosis.format_transfer import token_input_difference
    report = token_input_difference([1, 2, 3], [1, 2, 4, 5])
    assert report['first_mismatch'] == 2
    assert report['expected_ids'] == [1, 2, 3]
    assert report['actual_ids'] == [1, 2, 4, 5]
