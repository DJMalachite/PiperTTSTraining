"""Dataset report: split arithmetic, findings, and rendering.

The split feasibility figure is the one number that prevents a run failing with
"zero batches" an hour later, so it is checked directly rather than by eye.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from . import _support  # noqa: F401

from pipertrainer.dataset import report as R
from pipertrainer.dataset.metadata import Rejected, Row


def clips(count, seconds=5.0, chars=70, text="Hello there, friend of mine."):
    return [
        R.ClipStat(
            utt_id=f"{i:06d}", seconds=seconds, chars=chars, text=f"{text} {i}"
        )
        for i in range(1, count + 1)
    ]


def build(count=100, **kwargs):
    return R.build(
        [Row(f"{i:06d}.wav", "x.") for i in range(count)],
        kwargs.pop("stats", clips(count)),
        kwargs.pop("rejected", []),
        voice="testvoice",
        source="/tmp/rec.wav",
        source_duration=kwargs.pop("source_duration", 3600.0),
        **kwargs,
    )


class SplitTest(unittest.TestCase):
    def test_split_matches_the_trainers_arithmetic(self):
        report = build(100, validation_split=0.1, num_test_examples=5)
        split = report.split
        self.assertEqual((split.train, split.val, split.test), (85, 10, 5))

    def test_max_batch_size_is_reported(self):
        report = build(24, validation_split=0.1, num_test_examples=5)
        self.assertEqual(report.split.max_batch_size, 17)
        self.assertIn("max batch size", dict(
            (row[0], row[1]) for row in report.summary_rows()
        ))

    def test_oversized_batch_is_an_error_finding(self):
        report = build(24, batch_size=40)
        errors = [msg for level, msg in report.flags if level == "error"]
        self.assertTrue(any("zero batches" in m for m in errors), report.flags)

    def test_batch_size_within_the_split_is_not_flagged(self):
        report = build(100, batch_size=16)
        self.assertFalse(
            any("zero batches" in m for _, m in report.flags), report.flags
        )

    def test_split_section_appears_in_the_markdown(self):
        markdown = build(100).to_markdown()
        self.assertIn("## Training split", markdown)
        self.assertIn("train: **85**", markdown)


class FindingsTest(unittest.TestCase):
    def test_no_clips_is_an_error(self):
        report = build(0, stats=[])
        self.assertTrue(any(level == "error" for level, _ in report.flags))
        self.assertIn("no usable clips", report.flags[0][1])

    def test_tiny_dataset_is_an_error(self):
        # 20 clips x 5 s = 100 s of speech
        report = build(20, stats=clips(20))
        self.assertTrue(
            any("minutes of speech" in m for level, m in report.flags if level == "error"),
            report.flags,
        )

    def test_modest_dataset_is_a_warning_not_an_error(self):
        # 200 clips x 5 s = ~17 minutes
        report = build(200, stats=clips(200))
        messages = [m for level, m in report.flags if level == "warn"]
        self.assertTrue(any("low side" in m for m in messages), report.flags)
        self.assertFalse(any(level == "error" for level, _ in report.flags))

    def test_comfortable_dataset_has_no_size_finding(self):
        report = build(1000, stats=clips(1000))
        self.assertFalse(
            any("minutes of speech" in m for _, m in report.flags), report.flags
        )

    def test_short_clips_are_flagged(self):
        stats = clips(500) + [
            R.ClipStat("000501", seconds=0.5, chars=8, text="Yes now.")
        ]
        report = build(501, stats=stats, min_clip_floor=1.0)
        self.assertTrue(
            any("shorter than" in m for _, m in report.flags), report.flags
        )

    def test_empty_transcript_is_an_error(self):
        stats = clips(500) + [R.ClipStat("000501", seconds=3.0, chars=0, text="")]
        report = build(501, stats=stats)
        self.assertTrue(
            any("empty transcript" in m for level, m in report.flags if level == "error"),
            report.flags,
        )

    def test_low_character_rate_is_flagged(self):
        stats = clips(500) + [
            R.ClipStat("000501", seconds=10.0, chars=5, text="Hi.")
        ]
        report = build(501, stats=stats)
        self.assertTrue(
            any("little text" in m for _, m in report.flags), report.flags
        )

    def test_high_character_rate_is_flagged(self):
        stats = clips(500) + [
            R.ClipStat("000501", seconds=1.0, chars=200, text="x" * 200)
        ]
        report = build(501, stats=stats)
        self.assertTrue(
            any("much text" in m for _, m in report.flags), report.flags
        )

    def test_clipped_samples_are_flagged(self):
        stats = clips(500) + [
            R.ClipStat("000501", seconds=3.0, chars=40, text="Loud.", clipped_samples=500)
        ]
        report = build(501, stats=stats)
        self.assertTrue(any("clipped" in m for _, m in report.flags), report.flags)

    def test_duplicate_transcripts_are_flagged_as_a_repetition_loop(self):
        stats = clips(500) + [
            R.ClipStat(f"{i:06d}", seconds=3.0, chars=9, text="Thank you.")
            for i in range(600, 610)
        ]
        report = build(510, stats=stats)
        self.assertTrue(
            any("repetition loop" in m for _, m in report.flags), report.flags
        )

    def test_low_yield_is_flagged(self):
        # 500 clips x 5 s = 2500 s from a 4-hour source
        report = build(500, stats=clips(500), source_duration=14400.0)
        self.assertTrue(any("became" in m for _, m in report.flags), report.flags)

    def test_healthy_dataset_reports_nothing(self):
        report = build(1000, stats=clips(1000), source_duration=6000.0)
        self.assertEqual(report.flags, [], report.flags)
        self.assertIn("nothing to report", report.to_markdown())


class StatisticsTest(unittest.TestCase):
    def test_aggregates(self):
        stats = [
            R.ClipStat("1", seconds=2.0, chars=30, text="a"),
            R.ClipStat("2", seconds=4.0, chars=60, text="b"),
            R.ClipStat("3", seconds=6.0, chars=90, text="c"),
        ]
        report = build(3, stats=stats, source_duration=24.0)
        self.assertEqual(report.count, 3)
        self.assertEqual(report.total_seconds, 12.0)
        self.assertEqual(report.mean_seconds, 4.0)
        self.assertEqual(report.median_seconds, 4.0)
        self.assertEqual(report.shortest, 2.0)
        self.assertEqual(report.longest, 6.0)
        self.assertAlmostEqual(report.yield_fraction, 0.5)

    def test_chars_per_second(self):
        clip = R.ClipStat("1", seconds=4.0, chars=60, text="x")
        self.assertEqual(clip.chars_per_second, 15.0)

    def test_zero_duration_clip_reports_infinite_rate(self):
        clip = R.ClipStat("1", seconds=0.0, chars=10, text="x")
        self.assertEqual(clip.chars_per_second, float("inf"))

    def test_yield_with_no_source_duration_is_zero(self):
        report = build(3, stats=clips(3), source_duration=0.0)
        self.assertEqual(report.yield_fraction, 0.0)


class HistogramTest(unittest.TestCase):
    def test_histogram_has_one_line_per_bucket(self):
        stats = [
            R.ClipStat(str(i), seconds=1.0 + i * 0.5, chars=30, text="x")
            for i in range(24)
        ]
        report = build(24, stats=stats)
        lines = report.histogram(buckets=8)
        self.assertEqual(len(lines), 8)
        self.assertEqual(sum(int(line.split()[-1]) for line in lines), 24)

    def test_identical_durations_do_not_divide_by_zero(self):
        stats = [R.ClipStat(str(i), seconds=3.0, chars=30, text="x") for i in range(5)]
        report = build(5, stats=stats)
        self.assertEqual(len(report.histogram()), 1)

    def test_empty_report_has_no_histogram(self):
        self.assertEqual(build(0, stats=[]).histogram(), [])


class OutlierTest(unittest.TestCase):
    def test_worst_offender_is_first(self):
        stats = clips(50) + [
            R.ClipStat("000999", seconds=1.0, chars=300, text="x" * 300)
        ]
        report = build(51, stats=stats)
        self.assertEqual(report.outliers()[0].utt_id, "000999")

    def test_outliers_are_capped(self):
        report = build(200, stats=clips(200))
        self.assertLessEqual(len(report.outliers(limit=10)), 10)

    def test_outliers_appear_in_markdown_with_escaped_pipes(self):
        stats = clips(20) + [
            R.ClipStat("000999", seconds=1.0, chars=40, text="has | a pipe")
        ]
        markdown = build(21, stats=stats).to_markdown()
        self.assertIn("Clips worth listening to", markdown)
        self.assertNotIn("| has | a pipe |", markdown)
        self.assertIn("has / a pipe", markdown)


class RenderingTest(unittest.TestCase):
    def test_markdown_is_written_to_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.md"
            build(100).write(path)
            text = path.read_text(encoding="utf-8")
        self.assertIn("# Dataset report: testvoice", text)
        self.assertNotIn("\r\n", text)

    def test_rejected_reasons_are_summarised(self):
        rejected = [
            Rejected("1", "filler only ('uh')", "Uh."),
            Rejected("2", "filler only ('um')", "Um."),
            Rejected("3", "3.2 chars/s over 9.00 s — incomplete", "Hi."),
        ]
        markdown = build(100, rejected=rejected).to_markdown()
        self.assertIn("## Rejected", markdown)
        self.assertIn("| filler only | 2 |", markdown)

    def test_next_steps_mention_the_cache_consequence(self):
        # Editing metadata.csv changes piper's cache keys; the user should not be
        # surprised by a rebuild.
        markdown = build(100).to_markdown()
        self.assertIn("cache", markdown.lower())
        self.assertIn("./run train", markdown)


if __name__ == "__main__":
    unittest.main()
