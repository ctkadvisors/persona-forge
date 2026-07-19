from pathlib import Path

from personaforge.pack import card_system_prompt, exemplar_rows, load_pack

PACK_PATH = Path(__file__).resolve().parents[1] / "packs" / "camelot.json"


def test_load_camelot_pack():
    pack = load_pack(PACK_PATH)
    assert pack.world == "Arthurian legend"
    assert len(pack.cards) >= 6
    assert pack.card("Merlin").pronoun == "him"
    assert pack.card("Morgan le Fay").pronoun == "her"
    assert pack.scenarios and pack.provocations and pack.assignments and pack.boilerplate


def test_card_system_prompt_mentions_world_and_rules():
    pack = load_pack(PACK_PATH)
    sp = card_system_prompt(pack.card("Merlin"), pack.world)
    assert "Merlin" in sp and "Arthurian legend" in sp
    assert "Never break character" in sp


def test_exemplar_rows_validate():
    pack = load_pack(PACK_PATH)
    rows = exemplar_rows(pack)
    assert len(rows) == len(pack.exemplars)
    for row in rows:
        assert row["messages"][0]["role"] == "system"
        assert row["meta"]["kind"] == "roleplay"


def test_assignment_templates_format():
    pack = load_pack(PACK_PATH)
    card = pack.card("Guinevere")
    for tpl in pack.assignments:
        msg = tpl.format(name=card.name, pron=card.pronoun)
        assert "{" not in msg


def test_mythic_pack_validates():
    from personaforge.pack import load_pack
    p = load_pack("packs/mythic.json")
    assert len(p.cards) == 8 and len(p.scenarios) >= 12
    assert p.assignments and p.provocations and p.boilerplate


def test_exemplars_block():
    from personaforge.data.exemplars import EXEMPLARS, exemplar_block
    assert len(EXEMPLARS) >= 4
    block = exemplar_block(3)
    assert "elevated register" in block and "—" in block
