from personaforge.data.blend import blend_sft, split_train_val, write_datasets


def _row(u):
    return {"messages": [{"role": "user", "content": u},
                         {"role": "assistant", "content": "ok"}],
            "meta": {"source": "s", "kind": "chat"}}


def test_blend_caps_and_dedups():
    groups = {"a": [_row("x"), _row("x"), _row("y")], "b": [_row("z")]}
    out = blend_sft(groups, caps={"a": 5, "b": 5}, seed=1)
    contents = sorted(r["messages"][0]["content"] for r in out)
    assert contents == ["x", "y", "z"]


def test_split_and_write(tmp_path):
    rows = [_row(str(i)) for i in range(20)]
    tr, va = split_train_val(rows, 0.25, seed=1)
    assert len(va) == 5 and len(tr) == 15
    paths = write_datasets(tr, [], str(tmp_path))
    assert (tmp_path / "sft.jsonl").exists()
    assert paths["sft"].endswith("sft.jsonl")
