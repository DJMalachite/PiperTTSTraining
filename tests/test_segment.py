"""Grouping Whisper words into utterances.

Synthetic word lists only — no audio, no Whisper. This is the module that
decides dataset quality, so the bounds it promises (min/target/max, cuts in
silence, nothing silently lost) are all asserted directly.
"""

from __future__ import annotations

import unittest

from . import _support  # noqa: F401

from pipertrainer.dataset.segment import (
    Dropped,
    SegmentRules,
    Word,
    group_words,
    spans_to_utterances,
)

RULES = SegmentRules(
    min_seconds=1.0,
    target_seconds=8.0,
    max_seconds=14.0,
    boundary_gap=0.35,
    pad_before=0.15,
    pad_after=0.15,
)


def words(spec: str, *, start: float = 0.0, word: float = 0.4, gap: float = 0.05):
    """Build a word list from a compact spec.

    ``spec`` is whitespace-separated tokens; a token of ``|`` inserts a long
    pause (0.5 s) instead of a word, which is how utterance boundaries are
    expressed in these tests.
    """
    out: list[Word] = []
    clock = start
    for token in spec.split():
        if token == "|":
            clock += 0.5
            continue
        out.append(Word(text=token, start=clock, end=clock + word))
        clock += word + gap
    return out


def durations(utterances):
    return [round(u.speech_duration, 3) for u in utterances]


class BoundsTest(unittest.TestCase):
    def test_every_utterance_respects_min_and_max(self):
        # 60 words at 0.45 s each is ~27 s of speech with no pauses at all,
        # which is the hardest case for the splitter.
        result = group_words(words(" ".join(["word"] * 60)), RULES)
        self.assertTrue(result.utterances)
        for utterance in result.utterances:
            self.assertGreaterEqual(utterance.speech_duration, RULES.min_seconds)
            self.assertLessEqual(utterance.speech_duration, RULES.max_seconds)

    def test_sentence_boundaries_close_a_clip(self):
        result = group_words(words("one two three. four five six."), RULES)
        self.assertEqual(len(result.utterances), 2)
        self.assertEqual(result.utterances[0].text, "one two three.")
        self.assertEqual(result.utterances[1].text, "four five six.")

    def test_sentence_boundary_below_min_does_not_close(self):
        # "Hi." alone is 0.4 s, under the 1 s floor, so it must be merged.
        result = group_words(words("Hi. and then we continued talking."), RULES)
        self.assertEqual(len(result.utterances), 1)
        self.assertTrue(result.utterances[0].text.startswith("Hi."))

    def test_long_pause_closes_a_clip(self):
        result = group_words(words("one two three | four five six"), RULES)
        self.assertEqual(len(result.utterances), 2)

    def test_short_pause_does_not_close_a_clip(self):
        result = group_words(
            words("one two three four five six", gap=0.05), RULES
        )
        self.assertEqual(len(result.utterances), 1)

    def test_comma_closes_only_past_the_target(self):
        short = group_words(words("one two, three four."), RULES)
        self.assertEqual(len(short.utterances), 1)

        long_spec = " ".join(["word"] * 20) + ", " + " ".join(["word"] * 6) + "."
        long = group_words(words(long_spec), RULES)
        self.assertGreater(len(long.utterances), 1)

    def test_target_is_respected_when_boundaries_allow(self):
        # 40 words with a pause every 8 words: clips should land near target,
        # not at the ceiling.
        spec = " | ".join(" ".join(["word"] * 8) for _ in range(5))
        result = group_words(words(spec), RULES)
        for utterance in result.utterances:
            self.assertLess(utterance.speech_duration, RULES.target_seconds + 1.0)


class SplittingTest(unittest.TestCase):
    def test_overlong_group_is_split_recursively(self):
        # 100 words with no punctuation and no pauses: ~45 s.
        result = group_words(words(" ".join(["word"] * 100)), RULES)
        self.assertGreaterEqual(len(result.utterances), 4)
        for utterance in result.utterances:
            self.assertLessEqual(utterance.speech_duration, RULES.max_seconds)

    def test_split_prefers_the_widest_central_pause(self):
        # A 20-word run with one clear pause in the middle.
        left = words(" ".join(["a"] * 20))
        clock = left[-1].end + 1.2
        right = [
            Word(text="b", start=clock + i * 0.45, end=clock + i * 0.45 + 0.4)
            for i in range(20)
        ]
        result = group_words(left + right, RULES)
        # The first cut should fall in the 1.2 s pause, so no clip contains
        # both an 'a' and a 'b'.
        for utterance in result.utterances:
            self.assertFalse(
                "a" in utterance.text and "b" in utterance.text,
                f"clip spans the pause: {utterance.text!r}",
            )

    def test_split_produces_balanced_halves(self):
        result = group_words(words(" ".join(["word"] * 60)), RULES)
        spans = durations(result.utterances)
        # No clip should be a tiny shaving off the front.
        self.assertGreater(min(spans), RULES.min_seconds)

    def test_single_overlong_word_is_dropped_with_a_reason(self):
        glitched = [Word(text="hello", start=0.0, end=20.0)]
        result = group_words(glitched, RULES)
        self.assertEqual(result.utterances, [])
        self.assertEqual(len(result.dropped), 1)
        self.assertIn("timestamp glitch", result.dropped[0].reason)

    def test_overlong_word_among_good_ones_does_not_lose_the_rest(self):
        good = words("one two three four five.")
        glitch = [Word(text="zzz", start=100.0, end=130.0)]
        more = words("six seven eight nine ten.", start=200.0)
        result = group_words(good + glitch + more, RULES)
        self.assertEqual(len(result.utterances), 2)
        self.assertEqual(len(result.dropped), 1)


class MergingTest(unittest.TestCase):
    def test_short_fragment_merges_into_the_previous_clip(self):
        result = group_words(words("one two three four. yes."), RULES)
        self.assertEqual(len(result.utterances), 1)
        self.assertIn("yes.", result.utterances[0].text)

    def test_leading_short_fragment_merges_forward(self):
        result = group_words(words("Yes. then we carried on talking here."), RULES)
        self.assertEqual(len(result.utterances), 1)
        self.assertTrue(result.utterances[0].text.startswith("Yes."))

    def test_unmergeable_fragment_is_dropped_with_a_reason(self):
        # A lone short word separated from everything by huge gaps.
        lonely = [Word(text="hi", start=0.0, end=0.3)]
        result = group_words(lonely, RULES)
        self.assertEqual(result.utterances, [])
        self.assertEqual(len(result.dropped), 1)
        self.assertIn("below min_seconds", result.dropped[0].reason)

    def test_merging_never_exceeds_the_ceiling(self):
        spec = " ".join(["word"] * 30) + ". hi."
        result = group_words(words(spec), RULES)
        for utterance in result.utterances:
            self.assertLessEqual(utterance.speech_duration, RULES.max_seconds)


class CutPointTest(unittest.TestCase):
    def test_padding_is_applied(self):
        result = group_words(words("one two three four.", start=5.0), RULES)
        utterance = result.utterances[0]
        self.assertAlmostEqual(utterance.start, 5.0 - RULES.pad_before, places=3)
        self.assertAlmostEqual(
            utterance.end, utterance.word_end + RULES.pad_after, places=3
        )

    def test_start_never_goes_negative(self):
        result = group_words(words("one two three four.", start=0.0), RULES)
        self.assertGreaterEqual(result.utterances[0].start, 0.0)

    def test_cuts_land_at_the_pause_midpoint_and_never_overlap(self):
        # A 0.1 s gap is smaller than 2x padding, so the clamp must engage.
        first = words("one two three four.")
        clock = first[-1].end + 0.1
        second = [
            Word(text=t, start=clock + i * 0.45, end=clock + i * 0.45 + 0.4)
            for i, t in enumerate(["five", "six", "seven", "eight."])
        ]
        # Force two groups by making the gap a boundary.
        rules = SegmentRules(
            min_seconds=1.0,
            target_seconds=8.0,
            max_seconds=14.0,
            boundary_gap=0.05,
            pad_before=0.5,
            pad_after=0.5,
        )
        result = group_words(first + second, rules)
        self.assertEqual(len(result.utterances), 2)
        a, b = result.utterances
        self.assertLessEqual(
            a.end, b.start + 1e-9, "clips overlap; padding was not clamped"
        )

    def test_end_is_clamped_to_total_duration(self):
        result = group_words(
            words("one two three four."), RULES, total_duration=2.0
        )
        self.assertLessEqual(result.utterances[0].end, 2.0)


class HygieneTest(unittest.TestCase):
    def test_empty_input_yields_nothing(self):
        result = group_words([], RULES)
        self.assertEqual(result.utterances, [])
        self.assertEqual(result.dropped, [])

    def test_blank_and_zero_length_words_are_ignored(self):
        noisy = [
            Word(text="", start=0.0, end=0.5),
            Word(text="   ", start=0.5, end=1.0),
            Word(text="bad", start=2.0, end=2.0),
        ] + words("one two three four.", start=3.0)
        result = group_words(noisy, RULES)
        self.assertEqual(len(result.utterances), 1)
        self.assertEqual(result.utterances[0].text, "one two three four.")

    def test_unsorted_input_is_sorted(self):
        ordered = words("one two three four.")
        result = group_words(list(reversed(ordered)), RULES)
        self.assertEqual(result.utterances[0].text, "one two three four.")

    def test_no_speech_is_lost_without_an_explanation(self):
        # Every input word must end up in a clip or in `dropped`.
        source = (
            words("one two three four.")
            + [Word(text="glitch", start=50.0, end=90.0)]
            + words("five six seven eight.", start=100.0)
            + [Word(text="lonely", start=500.0, end=500.2)]
        )
        result = group_words(source, RULES)
        kept = " ".join(u.text for u in result.utterances)
        explained = " ".join(d.text for d in result.dropped)
        for word in source:
            token = word.text.strip()
            if not token:
                continue
            self.assertTrue(
                token in kept or token in explained,
                f"{token!r} vanished with no clip and no rejection reason",
            )

    def test_rules_reject_impossible_bounds(self):
        with self.assertRaises(ValueError):
            SegmentRules(min_seconds=5.0, target_seconds=2.0, max_seconds=10.0)
        with self.assertRaises(ValueError):
            SegmentRules(min_seconds=1.0, target_seconds=8.0, max_seconds=4.0)
        with self.assertRaises(ValueError):
            SegmentRules(min_seconds=0.0)

    def test_text_is_joined_with_single_spaces(self):
        result = group_words(words("one  two three four."), RULES)
        self.assertNotIn("  ", result.utterances[0].text)


class VadStrategyTest(unittest.TestCase):
    """The Whisper-independent path enforces the same bounds."""

    def test_spans_become_utterances(self):
        result = spans_to_utterances([(1.0, 4.0), (6.0, 9.0)], RULES)
        self.assertEqual(len(result.utterances), 2)
        for utterance in result.utterances:
            self.assertEqual(utterance.flags, ("vad",))
            self.assertEqual(utterance.text, "")

    def test_short_spans_are_dropped_with_a_reason(self):
        result = spans_to_utterances([(1.0, 1.3)], RULES)
        self.assertEqual(result.utterances, [])
        self.assertIn("below min_seconds", result.dropped[0].reason)

    def test_overlong_spans_are_chopped_within_bounds(self):
        result = spans_to_utterances([(0.0, 40.0)], RULES)
        self.assertGreater(len(result.utterances), 2)
        for utterance in result.utterances:
            self.assertLessEqual(utterance.speech_duration, RULES.max_seconds + 1e-9)
            self.assertGreaterEqual(
                utterance.speech_duration, RULES.min_seconds - 1e-9
            )

    def test_padding_and_clamping_apply(self):
        result = spans_to_utterances([(0.0, 3.0)], RULES, total_duration=3.1)
        self.assertGreaterEqual(result.utterances[0].start, 0.0)
        self.assertLessEqual(result.utterances[0].end, 3.1)

    def test_degenerate_spans_are_ignored(self):
        result = spans_to_utterances([(5.0, 5.0), (9.0, 8.0)], RULES)
        self.assertEqual(result.utterances, [])


if __name__ == "__main__":
    unittest.main()
