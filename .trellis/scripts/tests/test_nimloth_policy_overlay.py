from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


class NimlothPolicyOverlayTest(unittest.TestCase):
    def test_agents_is_a_thin_kernel_with_mandatory_routes(self) -> None:
        agents = text("AGENTS.md")
        self.assertLessEqual(len(agents.splitlines()), 60)
        for skill in (
            "on-experiment-start",
            "on-experiment-end",
            "slurm",
            "git-worktree",
            "memory",
            "on-progress",
        ):
            self.assertIn(skill, agents)
        for excluded_detail in (
            "heartbeat",
            ".trellis/.runtime",
            "[W-",
            "focused RED/GREEN",
            "fixed CoT",
            "CoT-conditioned",
            "cot-and-state.md",
        ):
            self.assertNotIn(excluded_detail, agents)
        self.assertLess(agents.index("[项目spec]"), agents.index("当前Trellis task"))

    def test_git_contract_uses_per_task_branches_and_one_canonical_slot(self) -> None:
        contract = text(".trellis/spec/governance/git-worktrees-and-protected-files.md")
        self.assertRegex(contract, r"(?i)every task|每个.{0,8}task")
        self.assertRegex(contract, r"(?i)task branch|task.{0,8}branch")
        self.assertRegex(contract, r"(?i)canonical.{0,80}(one|single|一个|单一)")
        self.assertRegex(contract, r"(?i)(parallel|并行).{0,80}worktree")

    def test_experiment_contract_is_remote_only_with_total_deadline(self) -> None:
        task_contract = text(".trellis/spec/experiments/task-contract.md")
        lifecycle = text(".trellis/spec/experiments/launch-and-lifecycle.md")
        combined = task_contract + "\n" + lifecycle
        self.assertRegex(combined, r"(?i)remote.only|remote-only|禁止.{0,12}本地")
        self.assertRegex(combined, r"(?i)(10|十).{0,24}(minute|分钟)")
        self.assertRegex(combined, r"(?i)(15|十五).{0,24}(minute|分钟).{0,40}(deadline|总)")
        self.assertRegex(combined, r"(?i)pending.{0,40}running|排队.{0,40}运行")
        self.assertRegex(combined, r"(?i)(cancel|取消).{0,80}(defer|blocker|阻断)")
        self.assertRegex(combined, r"10分钟.{0,80}(无需额外询问|without additional approval)")
        self.assertRegex(combined, r"超过10分钟.{0,80}(明确批准|explicit approval)")

    def test_task_creation_requires_upstream_consent(self) -> None:
        contract = text(".trellis/spec/governance/tasks-progress-and-memory.md")
        self.assertRegex(contract, r"(?i)(task.creation consent|task创建.{0,24}同意|创建task.{0,24}同意)")
        self.assertNotRegex(contract, r"明确.{0,20}实施.{0,40}授权.{0,20}创建")

    def test_project_skills_describe_their_safety_triggers(self) -> None:
        expected = {
            ".agents/skills/git-worktree/SKILL.md": ("worktree", "并行"),
            ".agents/skills/on-experiment-start/SKILL.md": ("实验", "启动"),
            ".agents/skills/on-experiment-end/SKILL.md": ("实验", "完成"),
            ".agents/skills/slurm/SKILL.md": ("Slurm", "远程"),
            ".agents/skills/memory/SKILL.md": ("memory", "批准"),
            ".agents/skills/on-progress/SKILL.md": ("work-item", "进展"),
        }
        for path, terms in expected.items():
            content = text(path)
            frontmatter = content.split("---", 2)[1]
            description = re.search(r"description:\s*(.*)", frontmatter)
            self.assertIsNotNone(description, path)
            for term in terms:
                self.assertIn(term.lower(), frontmatter.lower(), path)


if __name__ == "__main__":
    unittest.main()
