#!/usr/bin/env python3
"""Task-local, standard-library validation for the approved prompt rewrite."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TASK = ROOT / ".trellis/tasks/08-28-rewrite-trellis-prompts-chinese"
MANIFEST = json.loads((TASK / "research/scope-manifest.json").read_text(encoding="utf-8"))
FILES = [ROOT / item for item in MANIFEST["approvedPromptFiles"]]

EXPECTED_TAGS = [
    "no_task", "planning", "planning-inline", "in_progress",
    "in_progress-inline", "completed",
]
EXPECTED_HEADINGS = [
    "## Phase Index",
    "### Task threshold",
    "### Phase 1: Plan",
    "### Phase 2: Execute",
    "### Phase 3: Finish",
    "### Phase rules",
    "## Phase 1: Plan",
    "#### 1.0 Create task `[required · once]`",
    "#### 1.1 Requirements and risk exploration `[required · repeatable]`",
    "#### 1.2 Evidence research `[optional · repeatable]`",
    "#### 1.3 Configure selected context `[required · once for sub-agent platforms]`",
    "#### 1.4 Final planning review and activate `[required · once]`",
    "#### 1.5 Completion criteria",
    "## Phase 2: Execute",
    "#### 2.1 Implement `[required · repeatable]`",
    "#### 2.2 Quality check `[required · repeatable]`",
    "#### 2.3 Roll back or re-plan `[on demand]`",
    "## Phase 3: Finish",
    "#### 3.2 Debug retrospective `[on demand]`",
    "#### 3.3 Progress, memory, and spec review `[required · once]`",
    "#### 3.4 Complete-diff review and work commits `[required · once]`",
    "#### 3.5 Finish-work review and bookkeeping `[required · once]`",
    "## Platform consistency and upgrade boundary",
]
EXPECTED_NAMES = {
    "_template/SKILL.md": "your-skill-name",
    "git-worktree/SKILL.md": "git-worktree",
    "memory/SKILL.md": "memory",
    "on-experiment-start/SKILL.md": "on-experiment-start",
    "on-experiment-end/SKILL.md": "on-experiment-end",
    "on-progress/SKILL.md": "on-progress",
    "slurm/SKILL.md": "slurm",
}
HARD_RULES = {
    ".trellis/workflow.md": [
        "创建任务只批准规划",
        "实验启动还需要单独审批",
        "要求、范围、语义或授权发生变化时，必须返回规划阶段",
        "必须请求一次性的人类commit审批",
        "禁止amend、push、merge",
        "禁止手工编辑`.trellis/.template-hashes.json`或runtime session state",
        "有来源证据支持且可验证的工作",
    ],
    ".agents/skills/git-worktree/SKILL.md": [
        "禁止根据目录名推断branch",
        "每条会修改仓库的命令，都必须在同一次调用中",
        "未经针对已核验精确路径的明确批准，禁止使用`--force`",
    ],
    ".agents/skills/memory/SKILL.md": [
        "AI agent绝对禁止运行任何`./skill human ...`命令",
        "禁止手工编辑`.memory/memories.jsonl`或`.local/memory/memories.jsonl`",
        "依赖memory之前，必须运行`./skill memory get <id>`",
    ],
    ".agents/skills/on-experiment-start/SKILL.md": [
        "取得人类单独、明确的启动审批",
        "实施审批或任务启动审批不足以授权实验启动",
        "任一项目缺失，或获批后发生变化时，必须停止并重新询问",
        "每个必填字段都已明确、有来源证据支持并已核验",
    ],
    ".agents/skills/on-experiment-end/SKILL.md": [
        "一旦完成、失败、被取消或暂停，必须立即执行本skill",
        "禁止提升无效重试的结果",
        "只有实际使用过的memory经重新核验且确实有帮助时才upvote",
    ],
    ".agents/skills/on-progress/SKILL.md": [
        "必须暂停其他工作并立即执行本skill",
        "只能通过`memory` skill创建memory；禁止直接编辑JSONL",
        "禁止运行`./skill human ...`",
    ],
    ".agents/skills/slurm/SKILL.md": [
        "远程worktree必须指向该精确commit",
        "禁止直接在服务器上修改生产代码",
        "并取得单独的启动审批",
    ],
}
EXPECTED_FENCES = {
    ".trellis/workflow.md": [
        '''```bash\npython3 ./.trellis/scripts/task.py current --source\npython3 ./.trellis/scripts/task.py create "<title>" --slug <name>\npython3 ./.trellis/scripts/task.py start <task>\npython3 ./.trellis/scripts/task.py validate <task>\npython3 ./.trellis/scripts/get_context.py --mode packages\npython3 ./.trellis/scripts/get_context.py --mode phase --step <X.Y>\n```''',
        '''```text\nPhase 1: Plan    → classify risk, obtain task consent, research, persist and review artifacts\nPhase 2: Execute → implement approved scope, apply progress/experiment gates, verify repeatedly\nPhase 3: Finish  → full-scope check, memory/spec review, complete-diff review, work commits, wrap-up\n```''',
        '''```bash\npython3 ./.trellis/scripts/task.py create "<title>" --slug <name>\n```''',
        '''```bash\npython3 ./.trellis/scripts/task.py start <task>\n```''',
        '''```bash\ngit status --porcelain\ngit diff --stat\ngit diff --check\ngit log --oneline -5\n```''',
    ],
    ".agents/skills/git-worktree/SKILL.md": [
        '''```bash\nBRANCH="feat/my-feature"\nWT_DIR="../nimloth-$(printf '%s' "$BRANCH" | tr '/' '-')"\n\npwd\ngit status --short --branch\ngit worktree add -b "$BRANCH" "$WT_DIR" <approved-start-point>\n```''',
        '''```bash\nMAIN="../nimloth"  # replace with the actual shared-local-state worktree\ncd "$WT_DIR" && \\\n  ln -sfn "$MAIN/.local" .local\n```''',
        '''```bash\ncd "$WT_DIR" && \\\n  pwd && \\\n  git branch --show-current && \\\n  git status --short --branch && \\\n  test -f .local/SERVER.md && \\\n  test -f .agents/skills/git-worktree/SKILL.md && \\\n  test -f .agents/skills/slurm/SKILL.md\n```''',
        '''```bash\ngit worktree remove "$WT_DIR"\ngit worktree prune\n```''',
    ],
    ".agents/skills/memory/SKILL.md": [],
    ".agents/skills/slurm/SKILL.md": [
        '''```bash\n.local/scripts/query-resources.sh\n.local/scripts/query-resources.sh --only-free-gpu\n```''',
        '''```bash\nsrun --jobid <approved-job-id> --pty <command>\nsrun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w <allocated-node> bash -lc '<approved-command>'\n```''',
    ],
}

# Memory commands/examples are numerous; snapshot all fenced blocks from RED baseline.
EXPECTED_FENCES[".agents/skills/memory/SKILL.md"] = [
'''```bash
./skill memory add <title> <content>
./skill memory add --store local <title> <content>
./skill memory set <id> <field=value> [field=value ...]
./skill memory set --store local <id> <field=value> [field=value ...]
./skill memory search <keyword-regex> [--store all|repo|local] [--field all|title|content|evidence.filename|tags] [--tag TAG] [--level LEVEL] [--include-archived]
./skill memory get --store repo|local <id>
./skill memory upvote --store repo|local <id>
./skill memory human-verify --store repo|local <id>
```''',
'''```bash
./skill human memory-approve
./skill human memory-approve --store local
```''',
'''```bash
./skill memory add "Dataset split must be verified from loader metadata" "For Nimloth experiments, split names alone are not evidence; verify split semantics from the actual dataset/config/code path before launch."
./skill memory set M0001 'evidence=[{"filename":".trellis/spec/experiments/data-and-splits.md","line_start":1,"total_lines":18}]' 'tags=["experiments","data","split"]'
./skill memory human-verify M0001
```''',
'''```bash
./skill human memory-approve
```''',
]

PROSE_WORDS = re.compile(
    r"\b(?:the|and|before|after|must|never|only|when|with|from|for|use|read|"
    r"rules?|purpose|required|actions?|trigger|apply|confirm|present|obtain|"
    r"update|record|monitor|create|search|inspect|correct|approval|completion|"
    r"failure|cancelled|paused|unless|prefer|contains|remain|become|reviews?|"
    r"used|artifacts?|legacy|wrappers?|repository|memories|journals?|"
    r"source-backed|source-verified)\b",
    re.IGNORECASE,
)
ALLOWED_ENGLISH_HEADINGS = set(EXPECTED_HEADINGS)


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def fences(text: str) -> list[str]:
    return re.findall(r"```[^\n]*\n.*?```", text, flags=re.DOTALL)


def prose_lines(text: str) -> list[tuple[int, str]]:
    findings: list[tuple[int, str]] = []
    in_fence = False
    in_frontmatter = False
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if number == 1 and stripped == "---":
            in_frontmatter = True
            continue
        if in_frontmatter:
            if stripped == "---":
                in_frontmatter = False
                continue
            # Descriptions must be Chinese, while keys/ids stay English.
            if re.search(r"[A-Za-z]{3,}\s+[A-Za-z]{3,}", stripped) and not re.search(r"[\u4e00-\u9fff]", stripped):
                findings.append((number, line))
            continue
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not stripped or stripped in ALLOWED_ENGLISH_HEADINGS:
            continue
        # Remove inline code, Markdown destinations, and URLs before prose scan.
        candidate = re.sub(r"`[^`]+`", "", line)
        candidate = re.sub(r"\]\([^)]*\)", "]", candidate)
        candidate = re.sub(r"https?://\S+", "", candidate)
        if PROSE_WORDS.search(candidate):
            findings.append((number, line))
    return findings


def validate_markdown(path: Path, text: str) -> None:
    if text.count("```") % 2:
        fail(f"unbalanced code fences: {path.relative_to(ROOT)}")
    for destination in re.findall(r"\[[^]]*\]\(([^)]+)\)", text):
        if "://" in destination or destination.startswith("#"):
            continue
        target = (path.parent / destination.split("#", 1)[0]).resolve()
        if not target.exists():
            fail(f"missing relative link from {path.relative_to(ROOT)}: {destination}")


def main() -> None:
    if len(FILES) != 9 or len(set(FILES)) != 9:
        fail("scope manifest must contain exactly 9 unique prompt files")
    for path in FILES:
        if not path.is_file():
            fail(f"missing scope file: {path.relative_to(ROOT)}")
        validate_markdown(path, path.read_text(encoding="utf-8"))

    workflow = (ROOT / ".trellis/workflow.md").read_text(encoding="utf-8")
    for status in EXPECTED_TAGS:
        if workflow.count(f"[workflow-state:{status}]") != 1:
            fail(f"opening workflow tag changed: {status}")
        if workflow.count(f"[/workflow-state:{status}]") != 1:
            fail(f"closing workflow tag changed: {status}")
    workflow_lines = workflow.splitlines()
    for heading in EXPECTED_HEADINGS:
        if workflow_lines.count(heading) != 1:
            fail(f"parser/anchor heading changed: {heading}")
    for status in ("planning", "in_progress", "completed"):
        if status not in workflow:
            fail(f"task status missing: {status}")

    skills_root = ROOT / ".agents/skills"
    for rel, expected_name in EXPECTED_NAMES.items():
        text = (skills_root / rel).read_text(encoding="utf-8")
        match = re.match(r"---\n(.*?)\n---", text, flags=re.DOTALL)
        if not match:
            fail(f"invalid frontmatter: {rel}")
        frontmatter = match.group(1)
        if not re.search(rf"^name:\s*{re.escape(expected_name)}$", frontmatter, re.MULTILINE):
            fail(f"frontmatter name changed: {rel}")
        if not re.search(r"^description:\s*>-$", frontmatter, re.MULTILINE) and rel != "memory/SKILL.md":
            fail(f"frontmatter description key/style changed: {rel}")
        if rel == "memory/SKILL.md" and "description:" not in frontmatter:
            fail("memory frontmatter description key missing")

    for rel, expected in EXPECTED_FENCES.items():
        actual = fences((ROOT / rel).read_text(encoding="utf-8"))
        if actual != expected:
            fail(f"fenced command snapshot changed: {rel}")

    for rel, required_phrases in HARD_RULES.items():
        text = (ROOT / rel).read_text(encoding="utf-8")
        for phrase in required_phrases:
            if phrase not in text:
                fail(f"hard-rule assertion missing in {rel}: {phrase}")

    residuals: list[str] = []
    for path in FILES:
        for number, line in prose_lines(path.read_text(encoding="utf-8")):
            residuals.append(f"{path.relative_to(ROOT)}:{number}: {line}")
    if residuals:
        print("English natural-language candidates remain:", file=sys.stderr)
        print("\n".join(residuals), file=sys.stderr)
        raise SystemExit(1)

    print("PASS: 9-file scope manifest")
    print("PASS: workflow tags/status/parser headings")
    print("PASS: skill frontmatter ids/keys")
    print("PASS: fenced command snapshots")
    print("PASS: hard-rule semantic assertions")
    print("PASS: Markdown fences and relative links")
    print("PASS: English natural-language residual scan")


if __name__ == "__main__":
    main()
