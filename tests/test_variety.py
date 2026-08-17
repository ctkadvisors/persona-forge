from personaforge.eval.scorers import (
    cross_repetition, distinct_n, tale_variety)

# Same skeleton, nouns swapped -- the exact failure a shrunken model shows:
# voice and in-character stay perfect while every tale converges.
SAME_STORY = [
    "The old road wound through the grey hills, and the wind carried the voices of the dead.",
    "The old road wound through the grey fens, and the wind carried the voices of the lost.",
    "The old road wound through the grey woods, and the wind carried the voices of the slain.",
]
VARIED = [
    "Thorvald broke the ice of the Hjalmar with an antler pick and drew out a black fish.",
    "In the seventh year the bell tower at Ederwyn fell, and no mason would rebuild it.",
    "She counted nine ravens above the barley, then went indoors and barred the shutters.",
]


def test_cross_repetition_separates_same_story_from_varied():
    assert cross_repetition(SAME_STORY, 4) > cross_repetition(VARIED, 4)


def test_cross_repetition_zero_when_nothing_shared():
    assert cross_repetition(VARIED, 4) == 0.0


def test_cross_repetition_one_for_identical_texts():
    t = "the wind carried the voices of the dead across the grey hills"
    assert cross_repetition([t, t], 4) == 1.0


def test_distinct_n_lower_for_repetitive_set():
    assert distinct_n(SAME_STORY, 3) < distinct_n(VARIED, 3)


def test_distinct_n_is_one_when_no_ngram_repeats():
    assert distinct_n(["alpha beta gamma delta"], 2) == 1.0


def test_handles_degenerate_input():
    assert cross_repetition([], 4) == 0.0
    assert cross_repetition(["too short"], 4) == 0.0
    assert distinct_n([], 3) == 0.0


def test_tale_variety_report_shape_without_judge():
    r = tale_variety(VARIED)
    assert r["n_tales"] == 3
    assert set(r) == {"n_tales", "mean_words", "distinct_2", "distinct_3",
                      "cross_repetition_4"}
    assert r["mean_words"] > 0


def test_tale_variety_drops_empty_texts():
    assert tale_variety(["", "   ", VARIED[0]])["n_tales"] == 1


def test_tale_variety_uses_judge_when_given():
    class FakeJudge:
        def chat(self, *a, **k):
            return "0.8"
    r = tale_variety(VARIED, judge=FakeJudge())
    assert r["specificity_mean"] == 0.8
