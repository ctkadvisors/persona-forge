"""Regression tests for the name-identity checks. The original first-word
heuristic turned 'The Lady of the Lake' into the token 'the', which made
every reply look like an impersonation and silently dropped 100% of
assignment data on the first real multi-word-name pack."""
from personaforge.data.roleplay_synth import _names_correct
from personaforge.eval.scorers import named_correctly
from personaforge.pack import Card, distinctive_tokens, name_tokens, validate_pack, Pack

CARDS = [Card("Merlin", "a wizard", "cryptic"),
         Card("King Arthur", "the king", "noble"),
         Card("The Lady of the Lake", "keeper of Excalibur", "serene"),
         Card("Morgan le Fay", "an enchantress", "silken")]


def test_name_tokens_strips_articles():
    assert name_tokens("The Lady of the Lake") == {"lady", "lake"}
    assert name_tokens("King Arthur") == {"king", "arthur"}
    assert name_tokens("Morgan le Fay") == {"morgan", "fay"}


def test_distinctive_tokens_disjoint():
    arthur = next(c for c in CARDS if c.name == "King Arthur")
    assert distinctive_tokens(arthur, CARDS) == {"king", "arthur"}
    lady = next(c for c in CARDS if c.name == "The Lady of the Lake")
    assert distinctive_tokens(lady, CARDS) == {"lady", "lake"}


def test_the_does_not_trip_impersonation():
    merlin = CARDS[0]
    # regression: contains 'the' many times; must NOT read as the Lady
    assert _names_correct(merlin, "I am Merlin, the wisest of the wise, "
                                  "keeper of the old ways.", CARDS)


def test_actual_wrong_name_still_caught():
    merlin = CARDS[0]
    assert not _names_correct(merlin, "I am King Arthur of Camelot.", CARDS)
    assert not _names_correct(merlin, "The Lady of the Lake greets you.", CARDS)


def test_word_boundaries():
    merlin = CARDS[0]
    # 'gladly' contains 'lady' as a substring; must not trip
    assert _names_correct(merlin, "I shall gladly guide you, traveler.", CARDS)


def test_named_correctly_multiword():
    arthur = next(c for c in CARDS if c.name == "King Arthur")
    assert named_correctly(arthur, "I am Arthur, King of the Britons.", CARDS)
    assert not named_correctly(arthur, "I am Merlin the wizard.", CARDS)
    assert not named_correctly(arthur, "Greetings, traveler of the roads.", CARDS)


def test_validate_pack_catches_indistinct_names():
    pack = Pack(world="w",
                cards=[Card("King Arthur", "p", "s"), Card("King of the North", "p", "s")],
                scenarios=["s"], provocations=["a", "b", "c", "d"],
                assignments=["be {name}", "play {name}", "act as {name}", "you're {name}"],
                boilerplate=["x"])
    problems = validate_pack(pack)
    assert not any("distinctive" in p for p in problems)  # both have unique tokens
    pack2 = Pack(world="w",
                 cards=[Card("The King", "p", "s"), Card("King of Kings", "p", "s")],
                 scenarios=["s"], provocations=["a", "b", "c", "d"],
                 assignments=["be {name}", "play {name}", "act as {name}", "you're {name}"],
                 boilerplate=["x"])
    assert any("distinctive" in p for p in validate_pack(pack2))
