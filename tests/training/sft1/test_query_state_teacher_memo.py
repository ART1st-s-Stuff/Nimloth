from __future__ import annotations

import pytest
import torch

from nimloth.training.sft1.query_state_data import StrictQueryStateTeacherMemo


_SHA = "a" * 64
_DINO = "facebook/dinov2-large@revision:processor"


def test_teacher_memo_is_process_local_sha_and_dino_identity_keyed() -> None:
    memo = StrictQueryStateTeacherMemo(process_identity="fresh-process-a")
    calls = 0

    def compute() -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.arange(16 * 1024, dtype=torch.float32).reshape(16, 1024)

    first = memo.get_or_compute(
        original_image_sha256=_SHA,
        dino_identity=_DINO,
        compute=compute,
    )
    first.zero_()
    second = memo.get_or_compute(
        original_image_sha256=_SHA,
        dino_identity=_DINO,
        compute=compute,
    )
    assert calls == 1
    assert torch.count_nonzero(second) > 0
    assert second.device.type == "cpu"
    assert second.requires_grad is False
    assert memo.report().entries == 1
    assert memo.report().current_bytes == 16 * 1024 * 4
    assert memo.report().peak_bytes == memo.report().current_bytes


def test_teacher_memo_never_serializes_or_accepts_student_state() -> None:
    memo = StrictQueryStateTeacherMemo(process_identity="fresh-process-a")
    with pytest.raises(ValueError, match="SHA256"):
        memo.get_or_compute(
            original_image_sha256="/path/image.png",
            dino_identity=_DINO,
            compute=lambda: torch.zeros(16, 1024),
        )
    with pytest.raises(ValueError, match="detached.*CPU|target"):
        memo.get_or_compute(
            original_image_sha256=_SHA,
            dino_identity=_DINO,
            compute=lambda: torch.zeros(16, 2048, requires_grad=True),
        )
    with pytest.raises(RuntimeError, match="process-local|checkpoint"):
        memo.state_dict()


def test_teacher_memo_fresh_process_starts_empty_and_identity_cannot_change() -> None:
    one = StrictQueryStateTeacherMemo(process_identity="fresh-process-a")
    one.get_or_compute(
        original_image_sha256=_SHA,
        dino_identity=_DINO,
        compute=lambda: torch.zeros(16, 1024),
    )
    two = StrictQueryStateTeacherMemo(process_identity="fresh-process-b")
    assert two.report().entries == 0
    with pytest.raises(ValueError, match="DINO identity"):
        one.get_or_compute(
            original_image_sha256=_SHA,
            dino_identity="different",
            compute=lambda: torch.zeros(16, 1024),
        )
