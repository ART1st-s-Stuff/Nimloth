# Nimloth

Nimloth is a Python machine-learning project for building a **World Model Agent**.
## Current status

Project initialization is focused on AI collaboration prompts and rules.

## AI collaboration entry

All AI assistants must read [`AGENTS.md`](AGENTS.md). For durable memory, use the memory skill:

```bash
./skill memory search --store all <regex>
./skill memory get --store repo <id>
./skill memory add <title> <content>
./skill memory add --store local <title> <content>
./skill memory set --store repo <id> 'evidence=[{"filename":"...","line_start":1,"total_lines":10}]' 'tags=["..."]'
./skill memory upvote --store repo <id>
```

Human approval of AI-created memories:

```bash
./skill human memory-approve
./skill human memory-approve --store local
```

## Important note

Do not create code structure, model skeletons, training scripts, or experiments unless the human developer explicitly asks for them.
