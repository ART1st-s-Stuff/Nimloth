from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from nimloth.training.sft.stage1.loss import (
    weighted_answer_loss, training_loss, resolve_boundary_token_ids,
)
from nimloth.training.sft.stage1.checkpoint import objective_identities_match
from nimloth.training.sft.stage1.workflow import validate_workflow_resume


@pytest.mark.parametrize('action_weight,boundary_weight', [(1, 1), (16, 16), (1, 16), (8, 1)])
def test_protocol_loss_exact_shifted_gradient(action_weight, boundary_weight):
    torch.manual_seed(9)
    scores = torch.randn(2, 7, 16, requires_grad=True)
    reference = scores.detach().clone().requires_grad_()
    labels = torch.tensor([[-100, -100, 0, 8, 9, 10, 14], [-100, 15, 7, 9, -100, 10, 11]])
    loss = weighted_answer_loss(scores, labels, range(8), action_weight,
                                boundary_token_ids=(8, 9, 10), boundary_weight=boundary_weight, chunk_size=2)
    targets = labels[:, 1:]
    mask = targets != -100
    weights = torch.ones_like(targets, dtype=torch.float)
    weights[(targets >= 0) & (targets < 8)] = action_weight
    weights[(targets >= 8) & (targets <= 10)] = boundary_weight
    expected = (F.cross_entropy(reference[:, :-1].reshape(-1, 16), targets.reshape(-1), reduction='none').reshape_as(targets)[mask] * weights[mask]).sum() / weights[mask].sum()
    loss.backward()
    expected.backward()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(scores.grad, reference.grad)
    assert torch.count_nonzero(scores.grad[:, -1]) == 0
    assert torch.count_nonzero(scores.grad[0, 0]) == 0


def test_unweighted_model_path():
    model = lambda **batch: SimpleNamespace(loss=13)
    assert training_loss(model, {}) == 13


class Tokenizer:
    unk_token_id = 99
    eos_token_id = 10
    tokens = {**{f'<|action_({i})|>': i for i in range(8)}, '<|action_start|>': 8, '<|action_end|>': 9}
    def convert_tokens_to_ids(self, token):
        return self.tokens[token]
    def encode(self, token, **kwargs):
        return [self.tokens[token]]


def test_boundary_ids_require_valid_disjoint_eos():
    tok = Tokenizer()
    assert resolve_boundary_token_ids(tok) == (8, 9, 10)
    tok.eos_token_id = 7
    with pytest.raises(ValueError, match='disjoint'):
        resolve_boundary_token_ids(tok)


def test_resume_weight_change_warns_both_directions_but_dataset_rejected():
    old = {'stage': 'format', 'dataset': 'a', 'action_token_loss_weight': 8, 'action_token_loss_scope': 'action_number_tokens_v1'}
    new = {**old, 'action_token_loss_weight': 16, 'boundary_token_loss_weight': 16,
           'boundary_token_loss_scope': 'action_boundaries_and_eos_v1'}
    for a, b in [(old, new), (new, old)]:
        with pytest.warns(UserWarning, match='changed loss weights'):
            assert objective_identities_match(a, b)
    assert not objective_identities_match(old, {**new, 'dataset': 'b'})
    with pytest.raises(ValueError, match='unsupported'):
        objective_identities_match(old, {**new, 'action_token_loss_scope': 'unknown'})


def test_workflow_only_weight_overrides_allowed():
    old = {'overrides': ['--lr', '0.1', '--action-token-loss-weight', '8'], 'input_sha256': {'config': 'a'}}
    new = {**old, 'overrides': ['--lr', '0.1', '--action-token-loss-weight=16', '--boundary-token-loss-weight', '16']}
    with pytest.warns(UserWarning, match='loss overrides changed'):
        validate_workflow_resume(old, new)
    for bad in [{**new, 'input_sha256': {'config': 'b'}}, {**new, 'overrides': ['--lr', '0.2']}]:
        with pytest.raises(ValueError, match='parameters differ'):
            validate_workflow_resume(old, bad)


def test_validation_reuses_forward_and_excludes_padding():
    from nimloth.training.sft.stage1.trainer import evaluate, convergence_monitor
    torch.manual_seed(8)
    logits = torch.randn(1, 4, 16)
    labels = torch.tensor([[-100, 8, 2, 10]])
    class Model:
        calls = 0
        def eval(self): pass
        def train(self): pass
        def __call__(self, **batch):
            self.calls += 1
            return SimpleNamespace(logits=logits, loss=torch.tensor(float(self.calls)))
    model = Model()
    result = evaluate(model, [{'labels': labels}] * 3, torch.device('cpu'),
        include_batches=[True, False, True], action_token_ids=tuple(range(8)),
        boundary_token_ids=(8, 9, 10), action_weight=16, boundary_weight=16)
    assert model.calls == 3
    assert result['validation_lm_loss'] == 2
    expected = weighted_answer_loss(logits, labels, range(8), 16,
        boundary_token_ids=(8, 9, 10), boundary_weight=16)
    assert result['validation_weighted_lm_loss'] == pytest.approx(expected.item())
    assert convergence_monitor('format') == 'validation_lm_loss'


@pytest.mark.parametrize('weight', [float('nan'), float('inf'), 0.5])
def test_invalid_boundary_weight_rejected(weight):
    with pytest.raises(ValueError):
        weighted_answer_loss(torch.zeros(1, 3, 16), torch.tensor([[-100, 8, 10]]), range(8), 1,
            boundary_token_ids=(8, 9, 10), boundary_weight=weight)


def test_cli_and_yaml_boundary_weight(tmp_path):
    from nimloth.training.sft.stage1.cli import parse_args
    from nimloth.training.sft.stage1.config import sft1_yaml_defaults
    config = tmp_path / 'config.yaml'
    config.write_text('train:\n  boundary_token_loss_weight: 16\n')
    assert sft1_yaml_defaults(config)['boundary_token_loss_weight'] == 16
    argv = ['--model', str(tmp_path), '--train-jsonl', str(tmp_path / 'train'),
            '--val-jsonl', str(tmp_path / 'val'), '--output-dir', str(tmp_path / 'out'), '--epochs', '2']
    with pytest.raises(ValueError, match='only for format'):
        parse_args([*argv, '--dino-cache-root', str(tmp_path), '--boundary-token-loss-weight', '16'], stage='query')
    args, _ = parse_args([*argv, '--format-eval-jsonl', str(tmp_path / 'format'),
                       '--action-token-loss-weight', '16', '--boundary-token-loss-weight', '16'])
    assert args.action_token_loss_weight == args.boundary_token_loss_weight == 16


def test_query_identity_without_format_weight_scope():
    old = {'stage': 'query', 'action_token_loss_weight': 1, 'action_token_loss_scope': None}
    assert objective_identities_match(old, {**old, 'boundary_token_loss_weight': 1, 'boundary_token_loss_scope': None})
    with pytest.raises(ValueError, match='Stage 2'):
        objective_identities_match(old, {**old, 'boundary_token_loss_weight': 16,
                                       'boundary_token_loss_scope': None})


def test_weighted_boundary_identity_requires_declared_scope():
    identity = {'stage': 'format', 'action_token_loss_scope': 'action_number_tokens_v1',
                'boundary_token_loss_weight': 16}
    with pytest.raises(ValueError, match='lacks declared loss scope'):
        objective_identities_match(identity, identity)
