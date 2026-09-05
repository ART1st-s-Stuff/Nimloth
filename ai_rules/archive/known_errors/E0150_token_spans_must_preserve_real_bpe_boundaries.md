# E0150 — Token spans must preserve real BPE boundaries

## Error

ID192 attempt0 tokenized the instruction string in isolation and searched for
those IDs in the complete Qwen chat prompt. For some instructions, Qwen BPE
merges final punctuation with the following newline, so the isolated IDs are
not an exact subsequence. Job `529767` failed before feature extraction.

## Rule

When extracting a semantic text span from model hidden states, tokenize the
complete archived field with its actual prefix and suffix. Use tokenizer offset
mappings to select tokens overlapping the desired character span, then locate
the complete field IDs in the actual prompt. Do not repair mismatches by
silently dropping boundary tokens or using approximate string/token matches.
