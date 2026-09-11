"""Strict early-stage responses and explicit navigation service serialization."""
from __future__ import annotations

import re
from dataclasses import dataclass

from nimloth.agent.action_prompt import NAMES, TOKENS, format_action_prompt
from nimloth.latent.extraction import latent_state_tokens


@dataclass(frozen=True)
class EarlyProtocol:
    stage: str
    query_count: int | None = None
    query_mode: str | None = None

    @property
    def query_tokens(self) -> tuple[str, ...]:
        return latent_state_tokens(self.query_count) if self.query_count else ()

    def prompt(self, text: str) -> str:
        if self.stage == 'vagen':
            return text
        return format_action_prompt(text, latent_token_count=self.query_count)

    def parse(self, text: str) -> dict:
        # 全文匹配，禁止把截断、多动作或尾随垃圾修成可执行动作。
        def field(name: str) -> str:
            content = (
                r'(?:(?!</?(?:think|observation|reasoning|prediction|answer)>'
                r'|<\|(?:action_start|action_end|action_\([^|>]*\)'
                r'|latent_state(?:_\d+)?)\|>).)*'
            )
            return rf'<{name}>(?P<{name}>{content})</{name}>'

        thought = (
            r'<think>(?P<think>\s*'
            + field('observation')
            + r'\s*'
            + field('reasoning')
            + r'\s*'
            + field('prediction')
            + r'\s*)</think>'
        )
        if self.stage == 'vagen':
            tail = r'<answer>(?P<action>.*?)</answer>'
        else:
            tail = re.escape(''.join(self.query_tokens)) + r'<\|action_start\|>(?P<action><\|action_\([0-7]\)\|>)<\|action_end\|>'
        match = re.fullmatch(r'\s*' + thought + r'\s*' + tail + r'\s*', text, re.DOTALL)
        if match is None or any(
            not match[name].strip()
            for name in ('observation', 'reasoning', 'prediction')
        ):
            return {'format_correct': False, 'action_index': None, 'service_response': text if self.stage == 'vagen' else '',
                    'error': 'invalid_response_envelope'}
        action = match['action'].strip().lower() if self.stage == 'vagen' else match['action']
        names = NAMES if self.stage == 'vagen' else TOKENS
        if action not in names:
            return {'format_correct': False, 'action_index': None, 'service_response': text if self.stage == 'vagen' else '', 'error': 'invalid_action'}
        index = names.index(action)
        return {'format_correct': True, 'action_index': index, 'error': None,
                'service_response': text if self.stage == 'vagen' else '<think>' + match['think'] + '</think><answer>' + NAMES[index] + '</answer>'}
