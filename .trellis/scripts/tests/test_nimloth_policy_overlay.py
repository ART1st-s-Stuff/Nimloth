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
        self.assertLess(agents.index("[项目 spec]"), agents.index("经审查的 task"))

    def test_git_contract_uses_per_task_branches_and_one_canonical_slot(self) -> None:
        contract = text(".trellis/spec/governance/git-worktrees-and-protected-files.md")
        self.assertRegex(contract, r"每个修改仓库的 Trellis 任务")
        self.assertRegex(contract, r"不同任务不能共享同一实施分支")
        self.assertRegex(contract, r"每个主目录只提供一个主任务位置")
        self.assertRegex(contract, r"(?i)(parallel|并行).{0,80}worktree")

    def test_experiment_contract_is_remote_only_with_total_deadline(self) -> None:
        task_contract = text(".trellis/spec/experiments/task-contract.md")
        combined = task_contract
        self.assertRegex(combined, r"真实实验只能[^。\n]*远程环境执行")
        self.assertRegex(combined, r"(?i)(10|十).{0,24}(minute|分钟)")
        self.assertRegex(combined, r"15\s*分钟总截止时间")
        self.assertRegex(combined, r"(?i)pending.{0,40}running|排队.{0,40}运行")
        self.assertRegex(combined, r"(?i)(cancel|取消).{0,80}(defer|blocker|阻断)")
        self.assertRegex(combined, r"预计不超过 \*\*10 分钟\*\*[^。]*无需额外询问启动")
        self.assertRegex(combined, r"超过 10 分钟[^。]*明确批准")

    def test_task_creation_requires_upstream_consent(self) -> None:
        contract = text(".trellis/spec/governance/tasks-progress-and-memory.md")
        self.assertRegex(contract, r"没有活动任务时[^。]*先取得建任务的同意")
        self.assertIn("创建任务不代表实施已获批准", contract)

    def test_project_skills_describe_their_safety_triggers(self) -> None:
        expected = {
            ".agents/skills/git-worktree/SKILL.md": ("worktree", "创建"),
            ".agents/skills/on-experiment-start/SKILL.md": ("实验", "启动"),
            ".agents/skills/on-experiment-end/SKILL.md": ("实验", "完成"),
            ".agents/skills/slurm/SKILL.md": ("Slurm", "资源"),
            ".agents/skills/memory/SKILL.md": ("memory", "自主维护"),
            ".agents/skills/on-progress/SKILL.md": ("交接", "长 job"),
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
