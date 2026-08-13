"""Quality presets.

The upsample/hop invariant is the one upstream enforces with a bare
``ValueError("Upsample rates do not match hop length")``, so every preset is
checked against it here rather than an hour into a run.
"""

from __future__ import annotations

import unittest
from functools import reduce

from . import _support  # noqa: F401

from pipertrainer.train import presets


class PresetInvariantTest(unittest.TestCase):
    def test_every_preset_satisfies_the_hop_length_invariant(self):
        for name, preset in presets.PRESETS.items():
            with self.subTest(preset=name):
                product = reduce(
                    lambda a, b: a * b, preset["upsample_rates"], 1
                )
                self.assertEqual(
                    product,
                    preset["hop_length"],
                    f"{name}: prod(upsample_rates)={product} != hop_length="
                    f"{preset['hop_length']}",
                )

    def test_upsample_kernels_align_with_strides(self):
        for name, preset in presets.PRESETS.items():
            rates = preset["upsample_rates"]
            kernels = preset["upsample_kernel_sizes"]
            with self.subTest(preset=name):
                self.assertEqual(len(rates), len(kernels))
                for rate, kernel in zip(rates, kernels):
                    self.assertGreaterEqual(kernel, rate)
                    self.assertEqual(
                        (kernel - rate) % 2, 0, f"{name}: {kernel}-{rate} is odd"
                    )

    def test_resblock_lists_are_parallel(self):
        for name, preset in presets.PRESETS.items():
            with self.subTest(preset=name):
                self.assertEqual(
                    len(preset["resblock_kernel_sizes"]),
                    len(preset["resblock_dilation_sizes"]),
                )

    def test_resblock_is_a_string(self):
        # models.py does `resblock == "1"`; an int would silently select ResBlock2.
        for name, preset in presets.PRESETS.items():
            with self.subTest(preset=name):
                self.assertIsInstance(preset["resblock"], str)
                self.assertIn(preset["resblock"], ("1", "2"))

    def test_segment_size_default_divides_hop_length(self):
        for name, preset in presets.PRESETS.items():
            with self.subTest(preset=name):
                self.assertEqual(8192 % preset["hop_length"], 0)

    def test_every_preset_has_a_sample_rate_and_note(self):
        for name in presets.PRESETS:
            self.assertIn(name, presets.PRESET_SAMPLE_RATE)
            self.assertIn(name, presets.PRESET_NOTES)
            self.assertGreater(len(presets.PRESET_NOTES[name]), 40)


class PresetIdentityTest(unittest.TestCase):
    """Guards against a pin bump silently changing what 'medium' means."""

    UPSTREAM_DEFAULTS = {
        "resblock": "2",
        "resblock_kernel_sizes": [3, 5, 7],
        "resblock_dilation_sizes": [[1, 2], [2, 6], [3, 12]],
        "upsample_rates": [8, 8, 4],
        "upsample_initial_channel": 256,
        "upsample_kernel_sizes": [16, 16, 8],
        "filter_length": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "mel_channels": 80,
    }

    def test_medium_matches_upstream_vitsmodel_defaults(self):
        for key, expected in self.UPSTREAM_DEFAULTS.items():
            with self.subTest(key=key):
                self.assertEqual(presets.MEDIUM[key], expected)

    def test_low_is_medium_architecture_at_16khz(self):
        for key in self.UPSTREAM_DEFAULTS:
            self.assertEqual(presets.LOW[key], presets.MEDIUM[key], key)
        self.assertEqual(presets.PRESET_SAMPLE_RATE["low"], 16000)
        self.assertEqual(presets.PRESET_SAMPLE_RATE["medium"], 22050)

    def test_high_differs_in_the_documented_ways(self):
        self.assertEqual(presets.HIGH["resblock"], "1")
        self.assertEqual(presets.HIGH["upsample_initial_channel"], 512)
        self.assertEqual(presets.HIGH["upsample_rates"], [8, 8, 2, 2])
        self.assertEqual(presets.HIGH["hop_length"], presets.MEDIUM["hop_length"])

    def test_presets_are_copies_not_shared_references(self):
        # preset_for() must not let a caller mutate the module-level dict.
        first = presets.preset_for("medium")
        first["upsample_initial_channel"] = 999
        self.assertEqual(presets.preset_for("medium")["upsample_initial_channel"], 256)

    def test_unknown_quality_raises(self):
        with self.assertRaises(ValueError):
            presets.preset_for("ultra")


class TupleLiteralTest(unittest.TestCase):
    def test_nested_lists_render_as_python_tuples(self):
        self.assertEqual(presets._as_tuple_literal([8, 8, 4]), "(8,8,4)")
        self.assertEqual(
            presets._as_tuple_literal([[1, 2], [2, 6]]), "((1,2),(2,6))"
        )

    def test_strings_are_quoted(self):
        self.assertEqual(presets._as_tuple_literal("2"), '"2"')

    def test_hint_is_pasteable(self):
        hint = presets.high_quality_argv_hint()
        self.assertIn("--model.resblock \"1\"", hint)
        self.assertIn("--model.upsample_rates (8,8,2,2)", hint)


if __name__ == "__main__":
    unittest.main()
