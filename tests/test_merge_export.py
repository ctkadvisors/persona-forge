import json

from personaforge.merge_export import harden_chat_template

# Real tails from the Qwen3.5 family. The two sizes phrase the toggle the
# opposite way round, which is how a 4B shipped defaulting thinking ON.
TAIL_DEFAULT_ON = (
    "{%- if add_generation_prompt %}\n"
    "    {{- '<|im_start|>assistant\\n' }}\n"
    "    {%- if enable_thinking is defined and enable_thinking is false %}\n"
    "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
    "    {%- else %}\n"
    "        {{- '<think>\\n' }}\n"
    "    {%- endif %}\n"
    "{%- endif %}"
)
TAIL_DEFAULT_OFF = (
    "{%- if add_generation_prompt %}\n"
    "    {{- '<|im_start|>assistant\\n' }}\n"
    "    {%- if enable_thinking is defined and enable_thinking is true %}\n"
    "        {{- '<think>\\n' }}\n"
    "    {%- else %}\n"
    "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
    "    {%- endif %}\n"
    "{%- endif %}"
)


def _template(tmp_path):
    return (tmp_path / "chat_template.jinja").read_text(encoding="utf-8")


def test_hardens_template_that_defaults_thinking_on(tmp_path):
    (tmp_path / "chat_template.jinja").write_text(TAIL_DEFAULT_ON, encoding="utf-8")
    assert harden_chat_template(str(tmp_path)) == "hardened"
    out = _template(tmp_path)
    assert "enable_thinking" not in out
    assert "<think>\\n\\n</think>\\n\\n" in out


def test_hardens_template_that_defaults_thinking_off(tmp_path):
    """Defaulting off is not enough -- stacks force-inject enable_thinking."""
    (tmp_path / "chat_template.jinja").write_text(TAIL_DEFAULT_OFF, encoding="utf-8")
    assert harden_chat_template(str(tmp_path)) == "hardened"
    assert "enable_thinking" not in _template(tmp_path)


def test_writes_literal_backslash_n_not_real_newlines(tmp_path):
    """re.sub expands escapes in a plain replacement string; that would put raw
    0x0A bytes inside the jinja string literal instead of the two-character \\n."""
    (tmp_path / "chat_template.jinja").write_text(TAIL_DEFAULT_ON, encoding="utf-8")
    harden_chat_template(str(tmp_path))
    out = _template(tmp_path)
    emitted = out[out.index("<think>"):out.index("</think>") + len("</think>")]
    assert "\n" not in emitted, "raw newline leaked into the jinja string literal"
    assert emitted == "<think>\\n\\n</think>"


def test_respects_thinking_keep(tmp_path, monkeypatch):
    monkeypatch.setenv("THINKING", "keep")
    (tmp_path / "chat_template.jinja").write_text(TAIL_DEFAULT_ON, encoding="utf-8")
    assert harden_chat_template(str(tmp_path)) == "kept"
    assert _template(tmp_path) == TAIL_DEFAULT_ON


def test_handles_template_embedded_in_tokenizer_config(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": TAIL_DEFAULT_ON, "model_max_length": 8}), encoding="utf-8")
    assert harden_chat_template(str(tmp_path)) == "hardened"
    d = json.loads((tmp_path / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert "enable_thinking" not in d["chat_template"]
    assert d["model_max_length"] == 8, "unrelated keys must survive"


def test_no_toggle_is_left_alone(tmp_path):
    already = "{%- if add_generation_prompt %}\n    {{- '<think>\\n\\n</think>\\n\\n' }}\n{%- endif %}"
    (tmp_path / "chat_template.jinja").write_text(already, encoding="utf-8")
    assert harden_chat_template(str(tmp_path)) == "no-toggle"
    assert _template(tmp_path) == already
