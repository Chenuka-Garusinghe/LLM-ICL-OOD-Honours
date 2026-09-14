"""Chat-template wrapping for instruct models (Stage 1a).

v1 sent raw completion strings straight into vLLM (`self.llm.generate(prompts,
sampling_params)` in src/inference/llm_runner.py) even though both configured
models are `-Instruct` checkpoints, post-trained to expect their own chat
structure (`<|start_header_id|>...` for Llama, ChatML for Qwen).
`apply_chat_template` appeared nowhere in the codebase. Per Min et al. (lit
review ref [27]) demonstration *format* drives ICL behaviour as much as
content -- running an instruct model in raw-completion mode is exactly the
kind of format mismatch that would produce a large, uninterpretable prior
bias, which is what v1 measured (96-99% class-1 rate zero-shot). See
REDESIGN_RATIONALE.md §4.1/§5.1.
"""

from __future__ import annotations

from typing import Protocol


class SupportsChatTemplate(Protocol):
    def apply_chat_template(
        self, messages: list[dict[str, str]], tokenize: bool, add_generation_prompt: bool
    ) -> str: ...


class ChatFormatter:
    """Wraps a tokenizer's own chat template.

    `tokenizer` is duck-typed -- anything exposing `.apply_chat_template`
    with the standard HF signature works, whether it's a real
    `transformers.PreTrainedTokenizer`, an MLX tokenizer wrapper, or a test
    stub.
    """

    def __init__(self, tokenizer: SupportsChatTemplate):
        self.tokenizer = tokenizer

    def render(self, system: str, user: str) -> str:
        """Render a (system, user) message pair through the tokenizer's chat
        template, with the assistant generation prompt appended -- the
        model's next token is then its first response token, exactly what
        the existing `allowed_token_ids` constraint and `get_confidence` in
        src/inference/llm_runner.py expect (both already just consume a
        prompt string, unchanged).
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
