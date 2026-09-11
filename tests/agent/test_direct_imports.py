"""Direct prompt/protocol imports must not load optional planner dependencies."""
import os
import subprocess
import sys


def test_direct_imports_do_not_load_world_model():
    result = subprocess.run([sys.executable, "-c", "import sys; from nimloth.agent.action_prompt import format_action_prompt; from nimloth.rollout.early_records import summarize; import nimloth.training.sft.evaluation.cli; assert 'nimloth.wm' not in sys.modules; assert 'nimloth.agent.planning' not in sys.modules"], env=os.environ.copy(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_public_planner_import_is_preserved():
    from nimloth.agent import WorldModelPlanner
    from nimloth.agent.planning import WorldModelPlanner as implementation
    assert WorldModelPlanner is implementation
