from personaforge.decontam import Decontaminator

CORPUS = ("In a hole in the ground there lived a creature of comfort and habit, "
          "and it was fond of seed-cake and long afternoons beside the fire.")


def _d(n=8):
    return Decontaminator(CORPUS, ["Zarkon", "Velbrath"], n=n)


def test_ngram_overlap_detected_at_n():
    d = _d(n=8)
    ok, reason = d.check("He said that in a hole in the ground there lived a king.")
    assert not ok and reason == "ngram_overlap"


def test_no_overlap_below_n():
    d = _d(n=8)
    ok, reason = d.check("In a hole in the ground there dwelt someone else entirely.")
    assert ok and reason == "ok"


def test_blocklist_word_boundary():
    d = _d()
    ok, reason = d.check("The dark lord Zarkon rose again.")
    assert not ok and reason == "blocklist:zarkon"
    ok, _ = d.check("The zarkonite mineral is fictional but fine.")
    assert ok  # substring must not trigger


def test_blocklist_case_insensitive():
    ok, reason = _d().check("VELBRATH stirs beneath the ice.")
    assert not ok and reason == "blocklist:velbrath"


def test_report_counts():
    d = _d()
    d.check("A clean line of original prose about the sea.")
    d.check("The dark lord Zarkon rose again.")
    d.check("in a hole in the ground there lived a creature of legend")
    r = d.report()
    assert r["checked"] == 3 and r["ok"] == 1
    assert r["dropped"]["blocklist:zarkon"] == 1
    assert r["dropped"]["ngram_overlap"] == 1
