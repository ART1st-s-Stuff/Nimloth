from pathlib import Path
from types import SimpleNamespace
import pytest
from experiments.training.sft.stage3 import prepare_transition_cache as m


def test_new_output_and_exact_api(tmp_path,monkeypatch):
    calls=[]
    p=SimpleNamespace(image_processor=SimpleNamespace(),tokenizer=object())
    monkeypatch.setattr(m.AutoProcessor,'from_pretrained',lambda *a,**k:p)
    monkeypatch.setattr(m,'add_special_tokens',lambda *a,**k:None)
    monkeypatch.setattr(m,'build_compact_transition_preprocess_cache',lambda **k:calls.append(k))
    out=tmp_path/'cache'
    argv=['--source',str(tmp_path/'train.jsonl'),'--output',str(out),'--processor',str(tmp_path/'model'),'--latent-token-count','64','--max-length','12000','--max-pixels','100352']
    m.main(argv)
    assert calls[0]['success_only'] is False and calls[0]['force'] is False
    assert calls[0]['latent_token_count']==64 and calls[0]['mask_latent_query_labels'] is True
    with pytest.raises(FileExistsError): m.main(argv)
    assert len(calls)==1
