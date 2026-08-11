"""metadata.csv round-trips.

The critical property: anything we write must be read back identically by
piper's own reader, which is a *default-dialect* ``csv.reader(f, delimiter="|")``.
Writing with QUOTE_NONE would look tidier and silently corrupt any transcript
starting with a double quote, because upstream's reader would still treat that
quote as a quote character. So the test reads with upstream's exact call.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from . import _support  # noqa: F401

from pipertrainer.dataset import metadata as M


def upstream_read(path: Path) -> list[list[str]]:
    """Exactly how piper1-gpl reads metadata.csv (vits/dataset.py)."""
    with open(path, "r", encoding="utf-8") as handle:
        return [row for row in csv.reader(handle, delimiter="|") if row]


class RoundTripTest(unittest.TestCase):
    def round_trip(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.csv"
            M.write(rows, path)
            return upstream_read(path), M.read(path)

    def test_plain_rows(self):
        rows = [M.Row("000001", "Hello there."), M.Row("000002", "Goodbye.")]
        upstream, ours = self.round_trip(rows)
        self.assertEqual(upstream, [["000001", "Hello there."], ["000002", "Goodbye."]])
        self.assertEqual(ours, rows)

    def test_transcript_with_double_quotes_survives_upstreams_reader(self):
        # The case QUOTE_NONE would break.
        rows = [M.Row("000001", '"Hello," he said.')]
        upstream, ours = self.round_trip(rows)
        self.assertEqual(upstream[0][1], '"Hello," he said.')
        self.assertEqual(ours[0].text, '"Hello," he said.')

    def test_transcript_starting_with_a_quote(self):
        rows = [M.Row("000001", '"Stop!"')]
        upstream, ours = self.round_trip(rows)
        self.assertEqual(upstream[0][1], '"Stop!"')
        self.assertEqual(ours[0].text, '"Stop!"')

    def test_transcript_containing_the_delimiter_survives(self):
        # textnorm strips pipes, but the writer must not corrupt one if it slips
        # through: the csv module quotes the field and upstream unquotes it.
        rows = [M.Row("000001", "a|b")]
        upstream, ours = self.round_trip(rows)
        self.assertEqual(upstream[0], ["000001", "a|b"])
        self.assertEqual(ours[0].text, "a|b")

    def test_transcript_containing_a_newline_survives(self):
        rows = [M.Row("000001", "line one\nline two")]
        upstream, ours = self.round_trip(rows)
        self.assertEqual(upstream[0][1], "line one\nline two")
        self.assertEqual(ours[0].text, "line one\nline two")

    def test_unicode_survives(self):
        rows = [M.Row("000001", "café 你好 naïve.")]
        upstream, ours = self.round_trip(rows)
        self.assertEqual(upstream[0][1], "café 你好 naïve.")
        self.assertEqual(ours[0].text, "café 你好 naïve.")

    def test_multispeaker_rows(self):
        rows = [M.Row("000001", "Hello.", speaker="alice")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.csv"
            M.write(rows, path)
            self.assertEqual(upstream_read(path), [["000001", "alice", "Hello."]])
            self.assertEqual(M.read(path, multispeaker=True), rows)

    def test_file_uses_lf_endings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.csv"
            M.write([M.Row("1", "a."), M.Row("2", "b.")], path)
            raw = path.read_bytes()
        self.assertNotIn(b"\r\n", raw)
        self.assertEqual(raw.count(b"\n"), 2)


class OrderingTest(unittest.TestCase):
    """Row order is part of piper's cache key, so it must be stable."""

    def test_numeric_ids_sort_numerically_not_lexically(self):
        rows = [M.Row("10", "ten."), M.Row("2", "two."), M.Row("1", "one.")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.csv"
            M.write(rows, path)
            ids = [row.utt_id for row in M.read(path)]
        self.assertEqual(ids, ["1", "2", "10"])

    def test_zero_padded_ids_are_already_sortable(self):
        ids = [M.clip_id(i) for i in (1, 2, 10, 100)]
        self.assertEqual(ids, ["000001", "000002", "000010", "000100"])
        self.assertEqual(sorted(ids), ids)

    def test_prefix_is_applied(self):
        self.assertEqual(M.clip_id(7, prefix="sess1-"), "sess1-000007")

    def test_writing_twice_produces_identical_bytes(self):
        rows = [M.Row("000002", "b."), M.Row("000001", "a.")]
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.csv"
            second = Path(tmp) / "b.csv"
            M.write(rows, first)
            M.write(list(reversed(rows)), second)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_sorting_can_be_disabled(self):
        rows = [M.Row("000002", "b."), M.Row("000001", "a.")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.csv"
            M.write(rows, path, sort=False)
            self.assertEqual([r.utt_id for r in M.read(path)], ["000002", "000001"])


class ReadErrorTest(unittest.TestCase):
    def test_blank_lines_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.csv"
            path.write_text("1.wav|a.\n\n2.wav|b.\n", encoding="utf-8")
            self.assertEqual(len(M.read(path)), 2)

    def test_missing_text_column_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.csv"
            path.write_text("1.wav\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                M.read(path)

    def test_multispeaker_with_two_columns_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.csv"
            path.write_text("1.wav|hello.\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                M.read(path, multispeaker=True)

    def test_extra_columns_take_the_last_as_text(self):
        # piper reads text as row[-1] for the default dataset type.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.csv"
            path.write_text("1.wav|alice|hello.\n", encoding="utf-8")
            self.assertEqual(M.read(path)[0].text, "hello.")


class AudioResolutionTest(unittest.TestCase):
    """Must match piper's own lookup or our row count disagrees with training."""

    def test_exact_name_is_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            wavs = Path(tmp)
            (wavs / "000001.wav").write_bytes(b"")
            self.assertIsNotNone(M.resolve_audio(wavs, "000001.wav"))

    def test_extension_is_appended_when_missing(self):
        # piper retries with ".wav" appended, so a bare id is valid.
        with tempfile.TemporaryDirectory() as tmp:
            wavs = Path(tmp)
            (wavs / "000001.wav").write_bytes(b"")
            self.assertIsNotNone(M.resolve_audio(wavs, "000001"))

    def test_missing_file_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(M.resolve_audio(Path(tmp), "nope"))

    def test_count_usable_reports_missing_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wavs = root / "wavs"
            wavs.mkdir()
            (wavs / "000001.wav").write_bytes(b"")
            path = root / "metadata.csv"
            M.write(
                [M.Row("000001.wav", "a."), M.Row("000002.wav", "b.")], path
            )
            usable, missing = M.count_usable(path, wavs)
        self.assertEqual(usable, 1)
        self.assertEqual(missing, ["000002.wav"])


class RejectedTest(unittest.TestCase):
    def test_rejections_are_written_with_reasons(self):
        items = [
            M.Rejected("000003", "filler only ('uh')", "Uh.", 1.0, 1.4),
            M.Rejected("000004", "3.2 chars/s", "Hi.", 2.0, 5.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rejected.csv"
            count = M.write_rejected(items, path)
            with open(path, encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(count, 2)
        self.assertEqual(rows[0], ["utt_id", "start", "end", "reason", "text"])
        self.assertEqual(rows[1][0], "000003")
        self.assertIn("filler", rows[1][3])

    def test_empty_rejections_still_writes_a_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rejected.csv"
            M.write_rejected([], path)
            self.assertTrue(path.read_text(encoding="utf-8").startswith("utt_id"))


if __name__ == "__main__":
    unittest.main()
