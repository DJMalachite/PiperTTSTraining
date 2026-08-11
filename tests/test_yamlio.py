"""YAML round-trips, including the stdlib fallback parser.

The fallback exists for the bootstrap path (a fresh clone has no venv, so no
PyYAML), which means it has to be correct without ever being exercised in
normal use. These tests are that exercise: every case runs twice, once through
PyYAML if present and once with it forcibly disabled.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager

from . import _support  # noqa: F401  (path setup)

from pipertrainer import yamlio


@contextmanager
def no_pyyaml():
    saved = yamlio._pyyaml
    yamlio._pyyaml = None
    try:
        yield
    finally:
        yamlio._pyyaml = saved


class RoundTripTest(unittest.TestCase):
    CASES = [
        {"a": 1, "b": "two", "c": True, "d": None, "e": 1.5},
        {"nested": {"deep": {"deeper": "value"}}},
        {"seq": [1, 2, 3]},
        {"tuples": [[1, 2], [2, 6], [3, 12]]},
        {"empty_map": {}, "empty_seq": []},
        {"quoted": "has: colon", "hash": "trailing # hash", "pipe": "a|b"},
        {"numberish": "22050", "boolish": "true", "nullish": "null"},
        {"unicode": "caf\u00e9 na\u00efve \u4f60\u597d"},
        {"path": "/home/user/My Recordings/take 1.wav"},
        {"negative": -3, "exp": 2e-4, "zero": 0},
        {"mixed": {"list": [1, "two", False, None], "scalar": "x"}},
    ]

    def _round_trip(self, case):
        text = yamlio.dumps(case)
        return yamlio.loads(text)

    def test_round_trip_with_available_parser(self):
        for case in self.CASES:
            with self.subTest(case=case):
                self.assertEqual(self._round_trip(case), case)

    def test_round_trip_with_fallback_parser(self):
        with no_pyyaml():
            for case in self.CASES:
                with self.subTest(case=case):
                    self.assertEqual(self._round_trip(case), case)

    def test_parsers_agree(self):
        if yamlio._pyyaml is None:
            self.skipTest("PyYAML not installed; nothing to compare against")
        for case in self.CASES:
            text = yamlio.dumps(case)
            with no_pyyaml():
                mine = yamlio.loads(text)
            theirs = yamlio.loads(text)
            self.assertEqual(mine, theirs, f"parsers disagree on {case!r}")


class EmitterTest(unittest.TestCase):
    def test_comments_precede_keys(self):
        text = yamlio.dumps(
            {"voice": {"name": "x"}},
            comments={"voice": "About the voice.", "voice.name": "The name."},
        )
        lines = [line for line in text.splitlines() if line.strip()]
        self.assertEqual(lines[0], "# About the voice.")
        self.assertEqual(lines[1], "voice:")
        self.assertEqual(lines[2].strip(), "# The name.")
        self.assertEqual(lines[3].strip(), "name: x")

    def test_header_is_commented(self):
        text = yamlio.dumps({"a": 1}, header="line one\nline two")
        self.assertTrue(text.startswith("# line one\n# line two\n"))

    def test_comments_are_ignored_on_read(self):
        text = yamlio.dumps(
            {"a": 1, "b": "x"}, comments={"a": "note"}, header="hdr"
        )
        with no_pyyaml():
            self.assertEqual(yamlio.loads(text), {"a": 1, "b": "x"})

    def test_reserved_words_are_quoted(self):
        # A voice named "no" must not come back as False.
        text = yamlio.dumps({"name": "no", "other": "yes"})
        with no_pyyaml():
            self.assertEqual(yamlio.loads(text), {"name": "no", "other": "yes"})

    def test_long_nested_sequences_use_block_style(self):
        wide = [[i, i + 1, i + 2, i + 3, i + 4, i + 5] for i in range(12)]
        text = yamlio.dumps({"wide": wide})
        self.assertIn("\n  - [", text, "expected block style for a long sequence")
        with no_pyyaml():
            self.assertEqual(yamlio.loads(text)["wide"], wide)

    def test_non_finite_float_is_refused(self):
        with self.assertRaises(yamlio.YamlError):
            yamlio.dumps({"bad": float("nan")})


class FallbackParserTest(unittest.TestCase):
    """Cases the fallback must handle that the emitter never produces."""

    def parse(self, text):
        with no_pyyaml():
            return yamlio.loads(text)

    def test_inline_mapping(self):
        self.assertEqual(
            self.parse("env: { A: 1, B: two }"), {"env": {"A": 1, "B": "two"}}
        )

    def test_block_sequence_of_scalars(self):
        self.assertEqual(self.parse("xs:\n  - 1\n  - 2\n"), {"xs": [1, 2]})

    def test_single_quoted_strings_are_literal(self):
        self.assertEqual(self.parse(r"a: 'c:\path'"), {"a": r"c:\path"})

    def test_document_markers_are_skipped(self):
        self.assertEqual(self.parse("---\na: 1\n...\n"), {"a": 1})

    def test_blank_and_comment_lines_are_skipped(self):
        self.assertEqual(self.parse("# c\n\na: 1\n\n  # d\nb: 2\n"), {"a": 1, "b": 2})

    def test_hash_inside_quotes_is_not_a_comment(self):
        self.assertEqual(self.parse('a: "x # y"'), {"a": "x # y"})

    def test_key_with_no_value_is_null(self):
        self.assertEqual(self.parse("a:\nb: 1\n"), {"a": None, "b": 1})

    def test_malformed_line_raises(self):
        with self.assertRaises(yamlio.YamlError):
            self.parse("this is not yaml\n")


if __name__ == "__main__":
    unittest.main()
