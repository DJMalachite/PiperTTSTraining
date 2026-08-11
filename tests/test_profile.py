"""Profile schema: round-trips, coercion, migration, and metadata coverage.

The profile is the contract between the wizard, the YAML file, and argmap. If a
field can be written but not read back, or carries no help text, the wizard
silently degrades — so those properties are asserted here rather than left to
inspection.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from . import _support  # noqa: F401

from pipertrainer import profile as P
from pipertrainer import yamlio


class RoundTripTest(unittest.TestCase):
    def test_defaults_round_trip(self):
        original = P.Profile()
        restored, warnings = P.from_dict(P.to_dict(original))
        self.assertEqual(warnings, [])
        self.assertEqual(P.to_dict(restored), P.to_dict(original))

    def test_modified_values_round_trip(self):
        original = P.Profile()
        original.voice.name = "mariah"
        original.voice.quality = "high"
        original.audio.sample_rate = 16000
        original.data.batch_size = 6
        original.dataset.max_seconds = 10.5
        original.dataset.whisper.model = "large-v3"
        original.dataset.whisper.temperature = [0.0, 0.5]
        original.runtime.env = {"HSA_OVERRIDE_GFX_VERSION": "10.3.0"}
        original.model.extra = {"upsample_initial_channel": 384}

        restored, warnings = P.from_dict(P.to_dict(original))
        self.assertEqual(warnings, [])
        self.assertEqual(P.to_dict(restored), P.to_dict(original))
        self.assertEqual(restored.dataset.whisper.temperature, [0.0, 0.5])
        self.assertEqual(restored.runtime.env["HSA_OVERRIDE_GFX_VERSION"], "10.3.0")

    def test_round_trip_through_a_real_file(self):
        original = P.Profile()
        original.voice.name = "file-test"
        original.dataset.input_path = "/data/My Recordings/take 1.m4a"
        original.dataset.whisper.initial_prompt = "Piper, VITS, espeak-ng."
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.yaml"
            P.save(original, path)
            restored, warnings = P.load(path)
        self.assertEqual(warnings, [])
        self.assertEqual(P.to_dict(restored), P.to_dict(original))

    def test_saved_file_is_commented_and_reparses_without_pyyaml(self):
        saved_pyyaml = yamlio._pyyaml
        original = P.Profile()
        original.voice.name = "commented"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.yaml"
            P.save(original, path)
            text = path.read_text(encoding="utf-8")
            self.assertIn("# pipertrainer profile", text)
            self.assertGreater(
                sum(1 for line in text.splitlines() if line.strip().startswith("#")),
                40,
                "expected the schema help to be emitted as comments",
            )
            yamlio._pyyaml = None
            try:
                restored, warnings = P.load(path)
            finally:
                yamlio._pyyaml = saved_pyyaml
        self.assertEqual(warnings, [])
        self.assertEqual(P.to_dict(restored), P.to_dict(original))


class CoercionTest(unittest.TestCase):
    def test_numeric_strings_are_coerced(self):
        prof, warnings = P.from_dict(
            {"data": {"batch_size": "12", "validation_split": "0.2"}}
        )
        self.assertEqual(warnings, [])
        self.assertEqual(prof.data.batch_size, 12)
        self.assertIsInstance(prof.data.batch_size, int)
        self.assertAlmostEqual(prof.data.validation_split, 0.2)

    def test_yes_no_are_accepted_for_booleans(self):
        for text, expected in (("yes", True), ("no", False), ("on", True), ("0", False)):
            prof, _ = P.from_dict({"data": {"trim_silence": text}})
            self.assertIs(prof.data.trim_silence, expected, f"for {text!r}")

    def test_bad_boolean_raises(self):
        with self.assertRaises(P.ProfileError):
            P.from_dict({"data": {"trim_silence": "maybe"}})

    def test_bad_number_raises(self):
        with self.assertRaises(P.ProfileError):
            P.from_dict({"data": {"batch_size": "eight"}})

    def test_null_string_becomes_empty(self):
        # A commented-out value parses as None; it must not become "None".
        prof, _ = P.from_dict({"dataset": {"input_path": None}})
        self.assertEqual(prof.dataset.input_path, "")

    def test_unknown_keys_warn_but_do_not_fail(self):
        prof, warnings = P.from_dict(
            {"voice": {"name": "x", "nonsense": 1}, "bogus_section": {"a": 1}}
        )
        self.assertEqual(prof.voice.name, "x")
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("voice.nonsense" in w for w in warnings))
        self.assertTrue(any("bogus_section" in w for w in warnings))

    def test_non_mapping_top_level_raises(self):
        with self.assertRaises(P.ProfileError):
            P.from_dict(["not", "a", "mapping"])  # type: ignore[arg-type]


class BoundsValidationTest(unittest.TestCase):
    """A hand-edited profile with a nonsense value must be visible, not silent."""

    def test_out_of_range_value_warns_on_load(self):
        _, warnings = P.from_dict({"data": {"validation_split": 1.5}})
        self.assertTrue(
            any("above the supported maximum" in w for w in warnings), warnings
        )

    def test_below_minimum_warns(self):
        _, warnings = P.from_dict({"dataset": {"min_seconds": 0.2}})
        self.assertTrue(
            any("below the supported minimum" in w for w in warnings), warnings
        )

    def test_invalid_choice_warns(self):
        _, warnings = P.from_dict({"voice": {"quality": "ultra"}})
        self.assertTrue(any("is not one of" in w for w in warnings), warnings)

    def test_valid_profile_produces_no_warnings(self):
        self.assertEqual(P.validate(P.Profile()), [])

    def test_bounds_are_warnings_not_errors(self):
        # The value survives so the user can see and fix it; argmap is what
        # refuses to launch.
        prof, _ = P.from_dict({"data": {"validation_split": 1.5}})
        self.assertEqual(prof.data.validation_split, 1.5)


class MigrationTest(unittest.TestCase):
    def test_missing_schema_is_assumed_current(self):
        prof, warnings = P.from_dict({"voice": {"name": "old"}})
        self.assertEqual(warnings, [])
        self.assertEqual(prof.schema, P.SCHEMA_VERSION)

    def test_future_schema_is_refused(self):
        with self.assertRaises(P.ProfileError) as ctx:
            P.from_dict({"schema": P.SCHEMA_VERSION + 1})
        self.assertIn("newer than this tool", str(ctx.exception))

    def test_schema_key_is_not_reported_unknown(self):
        _, warnings = P.from_dict({"schema": P.SCHEMA_VERSION})
        self.assertEqual(warnings, [])


class SchemaHygieneTest(unittest.TestCase):
    """Properties the wizard and the YAML writer depend on."""

    def setUp(self):
        self.profile = P.Profile()
        self.specs = list(P.iter_specs(self.profile))

    def test_every_leaf_field_has_a_spec(self):
        # iter_specs only yields fields carrying a Spec, so compare against a
        # manual walk to catch a field declared without spec()/spec_factory().
        import dataclasses

        def walk(obj, trail=""):
            for f in dataclasses.fields(obj):
                value = getattr(obj, f.name)
                path = f"{trail}.{f.name}" if trail else f.name
                if dataclasses.is_dataclass(value):
                    yield from walk(value, path)
                else:
                    yield path, f

        missing = [
            path
            for path, f in walk(self.profile)
            if path != "schema" and "spec" not in f.metadata
        ]
        self.assertEqual(missing, [], f"fields without a Spec: {missing}")

    def test_specs_are_non_trivial(self):
        for path, spec, _ in self.specs:
            with self.subTest(path=path):
                self.assertGreater(len(spec.help), 20, f"{path}: help too short")
                self.assertIn(
                    spec.kind,
                    ("str", "int", "float", "bool", "path", "list", "dict", "choice"),
                    f"{path}: unknown kind {spec.kind!r}",
                )

    def test_choice_defaults_are_valid_choices(self):
        for path, spec, value in self.specs:
            if spec.choices and spec.kind in ("choice", "int", "str"):
                with self.subTest(path=path):
                    self.assertIn(
                        value, spec.choices, f"{path}: default {value!r} not a choice"
                    )

    def test_numeric_defaults_are_within_bounds(self):
        for path, spec, value in self.specs:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            with self.subTest(path=path):
                if spec.minimum is not None:
                    self.assertGreaterEqual(value, spec.minimum, path)
                if spec.maximum is not None:
                    self.assertLessEqual(value, spec.maximum, path)

    def test_every_spec_has_a_comment(self):
        comments = P.comment_map(self.profile)
        for path, _, _ in self.specs:
            self.assertIn(path, comments, f"{path} has no generated comment")
        for section in P.SECTION_HELP:
            self.assertIn(section, comments)

    def test_get_and_set_path_agree_with_iter_specs(self):
        for path, _, value in self.specs:
            self.assertEqual(P.get_path(self.profile, path), value, path)
        P.set_path(self.profile, "dataset.whisper.model", "tiny")
        self.assertEqual(self.profile.dataset.whisper.model, "tiny")

    def test_batch_size_lives_under_data_not_model(self):
        # piper links data.batch_size -> model.batch_size, and setting the
        # target side of a jsonargparse link is a hard error. The profile must
        # mirror the side the flag is emitted on.
        paths = {path for path, _, _ in self.specs}
        self.assertIn("data.batch_size", paths)
        self.assertNotIn("model.batch_size", paths)

    def test_sample_rate_has_exactly_one_home(self):
        paths = {path for path, _, _ in self.specs}
        self.assertIn("audio.sample_rate", paths)
        self.assertNotIn("model.sample_rate", paths)
        self.assertNotIn("data.sample_rate", paths)

    def test_min_seconds_floor_protects_against_silence_padding(self):
        # piper pads every clip up to segment_size; below that the model learns
        # silence. The schema must not allow a value under 1 s.
        spec = dict((p, s) for p, s, _ in self.specs)["dataset.min_seconds"]
        self.assertIsNotNone(spec.minimum)
        self.assertGreaterEqual(spec.minimum, 1.0)


class ActiveProfileTest(unittest.TestCase):
    def test_slug_is_used_for_filenames(self):
        from pipertrainer.paths import profile_path, slug

        self.assertEqual(slug("My Voice!"), "My-Voice")
        self.assertTrue(str(profile_path("My Voice!")).endswith("My-Voice.yaml"))

    def test_slug_never_returns_empty(self):
        from pipertrainer.paths import slug

        self.assertEqual(slug("///"), "voice")
        self.assertEqual(slug(""), "voice")


if __name__ == "__main__":
    unittest.main()
