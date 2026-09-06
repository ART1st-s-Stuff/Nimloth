"""验证唯一训练入口和生产代码与实验诊断之间的依赖边界。"""

import ast
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SFT = ROOT / "src/nimloth/training/sft"
STAGE3 = SFT / "stage3"
MODULES = sorted(
    ".".join(path.relative_to(STAGE3).with_suffix("").parts)
    for path in STAGE3.rglob("*.py")
    if path.name not in ("__init__.py", "__main__.py")
)


@pytest.mark.parametrize("suffix", MODULES)
def test_canonical_stage3_module_imports(suffix):
    module = importlib.import_module(f"nimloth.training.sft.stage3.{suffix}")
    assert Path(module.__file__).resolve().is_relative_to(STAGE3)


def test_retired_training_packages_have_no_implementation():
    for name in ("sft1", "sft2"):
        assert not list((SFT.parent / name).rglob("*.py"))


def test_production_training_does_not_depend_on_experiment_tools():
    for path in SFT.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            assert not any(module.startswith("experiments.") for module in modules), path


def test_stage3_keeps_historical_value_objective():
    from nimloth.training.common.value_semantics import SFT2_VALUE_OBJECTIVE
    from nimloth.training.sft.stage3.algorithm import SFT2_VALUE_OBJECTIVE as actual

    assert actual == SFT2_VALUE_OBJECTIVE == "decision_state_executed_action_mc_v3"
