import pytest
import torch
from experiments.training.sft.diagnosis.projector_probe import Answers, audit_records, metrics


def test_split_audit_rejects_seed_overlap_and_duplicate_ids():
    train = [{'id':'a','source_key':'a','seed':1}]
    val = [{'id':'b','source_key':'b','seed':2}]
    assert audit_records(train,val)['seed']['overlap'] == 0
    with pytest.raises(ValueError, match='seed'):
        audit_records(train,[dict(val[0],seed=1)])
    with pytest.raises(ValueError, match='duplicate id'):
        audit_records(train+train,val)


def test_constant_prediction_has_zero_observation_gain():
    target = torch.arange(48).reshape(6,2,4).float()
    prediction = target.mean(0).expand_as(target)
    result = metrics(prediction,target,target.mean(0),['a','a','b','b','c','c'])
    assert result['observation_gain'] == pytest.approx(0,abs=1e-5)
    assert result['observation_variance_ratio'] == 0


def test_perfect_prediction_has_pairing_advantage():
    target = torch.arange(48).reshape(6,2,4).float()
    result = metrics(target,target,target.mean(0),['a','a','b','b','c','c'])
    assert result['mse'] == 0
    assert result['observation_gain'] > 0
    assert result['wrong_pair_mse'] > 0
    assert result['observation_variance_ratio'] == pytest.approx(1)
    with pytest.raises(ValueError,match='impossible'):
        metrics(target,target,target.mean(0),['a']*4+['b']*2)


def test_cache_coverage_and_identity_fail_closed(tmp_path):
    (tmp_path/'train').mkdir()
    records = [{'id':'a','source_key':'source','seed':1}]
    with pytest.raises(ValueError,match='incomplete'):
        Answers(tmp_path,'train',records)
    torch.save({'states':torch.zeros(2,2,3),'targets':torch.zeros(2,2,4),'images':['x','y'],'id':'wrong','source_key':'source','seed':'1','index':0,'split':'train'},tmp_path/'train'/'000000.pt')
    with pytest.raises(ValueError,match='identity'):
        Answers(tmp_path,'train',records)


def test_scene_group_split_is_checked_independently_of_seed():
    train = [{'id':'a','source_key':'a','seed':1,'scene_id':'kitchen'}]
    val = [{'id':'b','source_key':'b','seed':2,'scene_id':'kitchen'}]
    with pytest.raises(ValueError,match='scene_id'):
        audit_records(train,val,'scene_id')
    val[0]['scene_id']='bedroom'
    result = audit_records(train,val,'scene_id')
    assert result['scene_id']['overlap'] == 0
    assert 'Qwen' in result['interpretation']


def test_production_aggregation_weights_answers_and_sampler_padding():
    from experiments.training.sft.diagnosis.projector_probe import aggregation
    result = aggregation(torch.tensor([1., 3., 9.]),torch.tensor([0,0,1]),2,3)
    assert result['answer_weighted_mse'] == pytest.approx(13/3)
    assert result['trajectory_weighted_mse'] == pytest.approx(5.5)
    assert result['production_padded_answer_mse'] == pytest.approx(17/5)
    assert result['padded_trajectory_count'] == 1


def test_production_baseline_gate_rejects_drift_before_training():
    from experiments.training.sft.diagnosis.projector_probe import check_production_baseline
    check_production_baseline(.735,.7349878549575806,.01)
    check_production_baseline(.9,None,.01)
    with pytest.raises(ValueError,match='before optimizer steps'):
        check_production_baseline(.75,.7349878549575806,.01)
    with pytest.raises(ValueError,match='finite positive'):
        check_production_baseline(float('nan'),.7349878549575806,.01)
