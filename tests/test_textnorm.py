"""Transcript cleaning and rejection."""

from __future__ import annotations

import unittest

from . import _support  # noqa: F401

from pipertrainer.dataset import textnorm as T

RULES = T.TextRules()


def norm(text, **overrides):
    rules = T.TextRules(**{**RULES.__dict__, **overrides}) if overrides else RULES
    return T.normalize(text, rules)


class CleaningTest(unittest.TestCase):
    def test_plain_text_passes_through(self):
        result = norm("Hello there, friend.")
        self.assertTrue(result.ok)
        self.assertEqual(result.text, "Hello there, friend.")

    def test_pipe_is_removed(self):
        # The metadata delimiter inside a transcript is always an artefact.
        result = norm("Hello|there.")
        self.assertTrue(result.ok)
        self.assertNotIn("|", result.text)
        self.assertEqual(result.text, "Hello there.")

    def test_whitespace_is_collapsed(self):
        self.assertEqual(norm("  Hello \n\t there.  ").text, "Hello there.")

    def test_control_characters_are_stripped(self):
        self.assertEqual(norm("Hel\x07lo there.").text, "Hel lo there.")

    def test_bracketed_annotations_are_dropped(self):
        for raw in ("[Music] Hello there.", "Hello there. (laughs)", "{inaudible} Hello there."):
            with self.subTest(raw=raw):
                result = norm(raw)
                self.assertTrue(result.ok)
                self.assertNotIn("[", result.text)
                self.assertNotIn("(", result.text)
                self.assertIn("Hello there", result.text)

    def test_bracketed_can_be_kept(self):
        result = norm("Hello (there).", drop_bracketed=False)
        self.assertIn("(there)", result.text)

    def test_curly_quotes_are_folded(self):
        result = norm("“Hello,” he said—quietly.")
        self.assertEqual(result.text, '"Hello," he said-quietly.')

    def test_quote_folding_can_be_disabled(self):
        result = norm("“Hello.”", normalize_quotes=False)
        self.assertIn("“", result.text)

    def test_unicode_is_nfc_normalised(self):
        decomposed = "café open."  # e + combining acute
        result = norm(decomposed)
        self.assertEqual(result.text, "café open.")

    def test_non_latin_text_survives(self):
        result = norm("你好世界。")
        self.assertTrue(result.ok)
        self.assertIn("你好", result.text)


class TerminalPunctuationTest(unittest.TestCase):
    def test_missing_terminal_punctuation_is_added(self):
        self.assertEqual(norm("Hello there").text, "Hello there.")

    def test_existing_terminal_punctuation_is_kept(self):
        for text in ("Really?", "Stop!", "Wait..."):
            self.assertEqual(norm(text).text, text)

    def test_ellipsis_is_folded_but_still_counts_as_terminal(self):
        # "…" folds to "..." for a smaller phoneme inventory; no extra full stop
        # should be appended on top.
        self.assertEqual(norm("Wait…").text, "Wait...")

    def test_trailing_comma_becomes_a_full_stop(self):
        self.assertEqual(norm("and then,").text, "and then.")

    def test_can_be_disabled(self):
        self.assertEqual(
            norm("Hello there", ensure_terminal_punctuation=False).text,
            "Hello there",
        )


class RejectionTest(unittest.TestCase):
    def test_empty_is_rejected(self):
        for raw in ("", "   ", None):
            with self.subTest(raw=raw):
                self.assertFalse(norm(raw).ok)

    def test_annotation_only_is_rejected(self):
        result = norm("[Music]")
        self.assertFalse(result.ok)
        self.assertIn("empty after cleaning", result.reason)

    def test_punctuation_only_is_rejected(self):
        result = norm("...")
        self.assertFalse(result.ok)
        self.assertIn("no letters", result.reason)

    def test_digits_only_is_rejected(self):
        # espeak would happily read "42", but a clip with no letters is almost
        # always a mis-segmentation.
        self.assertFalse(norm("42").ok)

    def test_filler_only_is_rejected(self):
        for raw in ("Uh.", "um", "Hmm...", "uh-huh"):
            with self.subTest(raw=raw):
                result = norm(raw)
                self.assertFalse(result.ok, raw)
                self.assertIn("filler", result.reason)

    def test_filler_inside_a_sentence_is_kept(self):
        self.assertTrue(norm("Um, I think so.").ok)

    def test_too_short_is_rejected(self):
        result = norm("A.", min_chars=3)
        self.assertFalse(result.ok)
        self.assertIn("min_chars", result.reason)

    def test_every_rejection_has_a_reason(self):
        for raw in ("", "[Music]", "...", "uh", "A"):
            result = norm(raw, min_chars=3)
            if not result.ok:
                self.assertTrue(result.reason, f"{raw!r} rejected with no reason")


class RateCheckTest(unittest.TestCase):
    """Characters per second catches misalignment the audio cannot show."""

    def test_normal_speech_passes(self):
        # ~15 chars/s is ordinary
        self.assertIsNone(T.check_rate("Hello there, friend." * 3, 4.0, RULES))

    def test_too_few_chars_means_a_missing_transcript(self):
        reason = T.check_rate("Hi.", 10.0, RULES)
        self.assertIsNotNone(reason)
        self.assertIn("incomplete", reason)

    def test_too_many_chars_means_borrowed_text(self):
        reason = T.check_rate("word " * 100, 2.0, RULES)
        self.assertIsNotNone(reason)
        self.assertIn("too long for the audio", reason)

    def test_zero_duration_is_infinite_rate(self):
        self.assertEqual(T.chars_per_second("abc", 0), float("inf"))
        self.assertIsNotNone(T.check_rate("abc", 0, RULES))

    def test_bounds_are_configurable(self):
        loose = T.TextRules(cps_min=0.1, cps_max=1000.0)
        self.assertIsNone(T.check_rate("Hi.", 10.0, loose))


class RepetitionTest(unittest.TestCase):
    """Whisper's repetition loop is a known failure on long audio."""

    def test_single_word_loop_is_detected(self):
        self.assertTrue(T.looks_repetitive("yes yes yes yes yes yes yes yes"))

    def test_phrase_loop_is_detected(self):
        self.assertTrue(
            T.looks_repetitive("thank you thank you thank you thank you thank you")
        )

    def test_ordinary_repetition_is_not_flagged(self):
        self.assertFalse(
            T.looks_repetitive("It was very very good, and I would go again.")
        )

    def test_short_text_is_never_flagged(self):
        self.assertFalse(T.looks_repetitive("yes yes"))


if __name__ == "__main__":
    unittest.main()
