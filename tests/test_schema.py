import pytest
from personaforge.schema import validate_sft_row, validate_dpo_row, write_jsonl, read_jsonl


def test_valid_sft_row_passes():
    validate_sft_row({
        "messages": [
            {"role": "system", "content": "You are a loremaster."},
            {"role": "user", "content": "Who is Elrond?"},
            {"role": "assistant", "content": "Lord of Rivendell."},
        ],
        "meta": {"source": "synth", "kind": "grounded_qa"},
    })


@pytest.mark.parametrize("bad", [
    {"messages": [], "meta": {"source": "x", "kind": "chat"}},
    {"messages": [{"role": "boss", "content": "hi"}], "meta": {"source": "x", "kind": "chat"}},
    {"messages": [{"role": "user", "content": "hi"}], "meta": {"source": "x", "kind": "bogus"}},
    {"messages": [{"role": "user"}], "meta": {"source": "x", "kind": "chat"}},
])
def test_invalid_sft_rows_raise(bad):
    with pytest.raises(ValueError):
        validate_sft_row(bad)


def test_valid_dpo_row_passes():
    validate_dpo_row({
        "prompt": [{"role": "user", "content": "Describe the sea."}],
        "chosen": "The grey sea sighed...",
        "rejected": "The sea is water.",
        "meta": {"source": "synth"},
    })


def test_jsonl_roundtrip(tmp_path):
    rows = [{"a": 1}, {"b": "two"}]
    p = tmp_path / "x.jsonl"
    assert write_jsonl(rows, str(p)) == 2
    assert read_jsonl(str(p)) == rows
