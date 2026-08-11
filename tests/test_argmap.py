"""Profile -> ``piper.train fit`` translation.

The golden test is the important one: it pins the exact config we hand to
upstream, so bumping the piper1-gpl pin in pins.toml fails here rather than
three hours into a run.

The negative tests encode upstream behaviours that are expensive to rediscover:
link-argument targets, gradient clipping under manual optimization, the
zero-batches trap, and the strict-load architecture check.
"""

from __future__ import annotations

import unittest

from . import _support  # noqa: F401

from pipertrainer import profile as P
from pipertrainer.paths import VoicePaths
from pipertrainer.train import argmap, presets


def make_profile(**overrides) -> P.Profile:
    prof = P.Profile()
    prof.voice.name = "testvoice"
    prof.finetune.mode = "none"
    for dotted, value in overrides.items():
        P.set_path(prof, dotted.replace("__", "."), value)
    return prof


class GoldenConfigTest(unittest.TestCase):
    """Pins the emitted config for a default medium profile."""

    def setUp(self):
        self.plan = argmap.build(make_profile(), paths=VoicePaths("testvoice"))

    def test_sections_present(self):
        self.assertEqual(
            set(self.plan.config), {"seed_everything", "trainer", "model", "data"}
        )

    def test_model_block(self):
        expected = {
            # architecture, from the medium preset
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
            "mel_fmin": 0.0,
            "mel_fmax": None,
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0.1,
            # from the profile
            "sample_rate": 22050,
            "segment_size": 8192,
            "num_speakers": 1,
            "learning_rate": 2e-4,
            "learning_rate_d": 1e-4,
            "lr_decay": 0.999875,
            "lr_decay_d": 0.9999,
            "warmup_epochs": 0,
            "c_mel": 45,
            "c_kl": 1.0,
            "use_sdp": True,
            "use_mrd": False,
            "mos_metric": "utmos",
        }
        self.assertEqual(self.plan.model, expected)

    def test_data_block(self):
        data = self.plan.data
        paths = VoicePaths("testvoice")
        self.assertEqual(data["voice_name"], "testvoice")
        self.assertEqual(data["csv_path"], str(paths.metadata_csv))
        self.assertEqual(data["audio_dir"], str(paths.wavs))
        self.assertEqual(data["cache_dir"], str(paths.cache))
        self.assertEqual(data["config_path"], str(paths.piper_config_json))
        self.assertEqual(data["espeak_voice"], "en-us")
        self.assertEqual(data["batch_size"], 16)
        self.assertEqual(data["num_symbols"], 256)
        self.assertEqual(data["phoneme_type"], "espeak")
        self.assertEqual(data["dataset_type"], "text")
        self.assertNotIn("phonemes_path", data)

    def test_audio_dir_points_at_wavs_not_its_parent(self):
        # We emit bare NNNNNN.wav in metadata.csv, so audio_dir is the wavs
        # directory itself. The legacy dataset script wrote "wavs/N.wav", which
        # forced audio_dir to be the parent -- a classic footgun.
        self.assertTrue(self.plan.data["audio_dir"].endswith("wavs"))

    def test_trainer_block(self):
        trainer = self.plan.trainer
        self.assertEqual(trainer["precision"], "32-true")
        self.assertEqual(trainer["max_epochs"], -1)
        self.assertEqual(trainer["accelerator"], "auto")
        self.assertEqual(trainer["devices"], "auto")
        self.assertEqual(
            trainer["default_root_dir"], str(VoicePaths("testvoice").run_root)
        )
        # Upstream's default save_top_k means we must NOT emit callbacks, since
        # setting trainer.callbacks replaces trainer_defaults wholesale.
        self.assertNotIn("callbacks", trainer)
        self.assertNotIn("accumulate_grad_batches", trainer)
        self.assertNotIn("max_steps", trainer)

    def test_no_link_target_is_ever_emitted(self):
        for source, target in argmap.LINKS:
            section, key = target.split(".")
            with self.subTest(target=target):
                self.assertNotIn(
                    key,
                    self.plan.config[section],
                    f"{target} is a jsonargparse link target computed from {source}",
                )

    def test_link_sources_are_emitted_on_the_right_side(self):
        self.assertIn("sample_rate", self.plan.model)
        self.assertIn("batch_size", self.plan.data)
        self.assertIn("num_symbols", self.plan.data)
        self.assertIn("num_speakers", self.plan.model)
        for key in ("filter_length", "hop_length", "win_length", "segment_size"):
            self.assertIn(key, self.plan.model, key)

    def test_no_blocked_key_is_ever_emitted(self):
        for blocked in argmap.BLOCKED:
            section, key = blocked.split(".")
            self.assertNotIn(key, self.plan.config.get(section, {}), blocked)

    def test_argv_is_reproducible_by_hand(self):
        argv = self.plan.argv
        self.assertEqual(argv[0], "fit")
        joined = " ".join(argv)
        self.assertIn("--model.sample_rate 22050", joined)
        self.assertIn("--data.batch_size 16", joined)
        self.assertIn("--model.upsample_rates '(8,8,4)'", joined)
        self.assertIn("--model.resblock 2", joined)
        self.assertIn("--model.mel_fmax null", joined)
        self.assertIn("--model.use_sdp true", joined)

    def test_config_survives_a_yaml_round_trip(self):
        from pipertrainer import yamlio

        text = yamlio.dumps(self.plan.config)
        self.assertEqual(yamlio.loads(text), self.plan.config)


class ForbiddenKeyTest(unittest.TestCase):
    def test_model_extra_cannot_set_a_link_target(self):
        prof = make_profile()
        prof.model.extra = {"batch_size": 8}
        with self.assertRaises(argmap.ArgMapError) as ctx:
            argmap.build(prof)
        self.assertIn("data.batch_size instead", str(ctx.exception))

    def test_model_extra_cannot_set_a_blocked_key(self):
        prof = make_profile()
        prof.model.extra = {"grad_clip": 5.0}
        with self.assertRaises(argmap.ArgMapError) as ctx:
            argmap.build(prof)
        self.assertIn("never applied", str(ctx.exception))

    def test_extra_argv_rejects_link_targets(self):
        for target in ("data.sample_rate", "model.batch_size", "data.hop_length"):
            with self.subTest(target=target):
                with self.assertRaises(argmap.ArgMapError):
                    argmap.build(make_profile(), extra_argv=[f"--{target}", "1"])

    def test_extra_argv_rejects_gradient_clipping(self):
        with self.assertRaises(argmap.ArgMapError) as ctx:
            argmap.build(make_profile(), extra_argv=["--trainer.gradient_clip_val=5"])
        self.assertIn("manual optimization", str(ctx.exception))

    def test_extra_argv_allows_ordinary_flags(self):
        plan = argmap.build(
            make_profile(), extra_argv=["--trainer.limit_train_batches", "2"]
        )
        self.assertIn("--trainer.limit_train_batches", plan.argv)

    def test_accumulate_grad_batches_warns_about_being_inert(self):
        plan = argmap.build(make_profile(trainer__accumulate_grad_batches=2))
        self.assertEqual(plan.trainer["accumulate_grad_batches"], 2)
        self.assertTrue(
            any("manual optimization" in w for w in plan.warnings),
            plan.warnings,
        )


class ArchitectureInvariantTest(unittest.TestCase):
    def test_hop_length_mismatch_is_refused_with_a_fix(self):
        prof = make_profile()
        prof.model.extra = {"upsample_rates": [8, 8, 2]}  # 128, not 256
        with self.assertRaises(argmap.ArgMapError) as ctx:
            argmap.build(prof)
        message = str(ctx.exception)
        self.assertIn("multiply to 128", message)
        self.assertIn("hop_length is 256", message)
        self.assertIn("(8,8,4)", message)

    def test_mismatched_kernel_count_is_refused(self):
        prof = make_profile()
        prof.model.extra = {"upsample_kernel_sizes": [16, 16]}
        with self.assertRaises(argmap.ArgMapError) as ctx:
            argmap.build(prof)
        self.assertIn("same length", str(ctx.exception))

    def test_kernel_smaller_than_stride_is_refused(self):
        prof = make_profile()
        prof.model.extra = {"upsample_kernel_sizes": [16, 16, 2]}
        with self.assertRaises(argmap.ArgMapError) as ctx:
            argmap.build(prof)
        self.assertIn("smaller than", str(ctx.exception))

    def test_odd_kernel_stride_difference_is_refused(self):
        prof = make_profile()
        prof.model.extra = {"upsample_kernel_sizes": [16, 16, 9]}
        with self.assertRaises(argmap.ArgMapError) as ctx:
            argmap.build(prof)
        self.assertIn("odd", str(ctx.exception))

    def test_mismatched_resblock_lists_are_refused(self):
        prof = make_profile()
        prof.model.extra = {"resblock_kernel_sizes": [3, 5]}
        with self.assertRaises(argmap.ArgMapError) as ctx:
            argmap.build(prof)
        self.assertIn("its own dilation list", str(ctx.exception))

    def test_segment_size_must_divide_hop_length(self):
        with self.assertRaises(argmap.ArgMapError) as ctx:
            argmap.build(make_profile(model__segment_size=8000))
        self.assertIn("not a multiple of hop_length", str(ctx.exception))
        self.assertIn("7936", str(ctx.exception))  # suggested nearest value

    def test_halved_segment_size_is_accepted(self):
        plan = argmap.build(make_profile(model__segment_size=4096))
        self.assertEqual(plan.model["segment_size"], 4096)

    def test_bad_resblock_value_is_refused(self):
        prof = make_profile()
        prof.model.extra = {"resblock": 3}
        with self.assertRaises(argmap.ArgMapError) as ctx:
            argmap.build(prof)
        self.assertIn("must be the string", str(ctx.exception))

    def test_high_preset_passes_every_invariant(self):
        plan = argmap.build(make_profile(voice__quality="high"))
        self.assertEqual(plan.model["upsample_rates"], [8, 8, 2, 2])
        self.assertEqual(plan.model["upsample_initial_channel"], 512)
        self.assertEqual(plan.model["hop_length"], 256)

    def test_low_preset_warns_when_sample_rate_was_not_lowered(self):
        plan = argmap.build(make_profile(voice__quality="low"))
        self.assertTrue(
            any("16000 Hz" in w for w in plan.warnings), plan.warnings
        )

    def test_low_preset_at_16khz_is_clean(self):
        plan = argmap.build(
            make_profile(voice__quality="low", audio__sample_rate=16000)
        )
        rate_warnings = [w for w in plan.warnings if "sample_rate" in w]
        self.assertEqual(rate_warnings, [])
        self.assertEqual(plan.model["sample_rate"], 16000)


class SplitArithmeticTest(unittest.TestCase):
    """Mirrors VitsDataModule.setup exactly; drop_last makes this load-bearing."""

    def test_split_matches_upstream_formula(self):
        split = argmap.split_sizes(100, 0.1, 5)
        self.assertEqual((split.train, split.val, split.test), (85, 10, 5))
        self.assertEqual(split.total, 100)

    def test_test_examples_are_clamped_on_tiny_datasets(self):
        split = argmap.split_sizes(3, 0.1, 5)
        # valid = int(3*0.1) = 0; test = min(5, max(0, 3-0-1)) = 2; train = 1
        self.assertEqual((split.train, split.val, split.test), (1, 0, 2))

    def test_empty_dataset_is_refused(self):
        with self.assertRaises(argmap.ArgMapError) as ctx:
            argmap.check_dataset_math(0, 0.1, 5, 1)
        self.assertIn("no usable utterances", str(ctx.exception))

    def test_upstream_formula_always_leaves_one_training_example(self):
        # test_examples are clamped to total-valid-1, so train >= 1 for any
        # validation_split the schema permits. Worth pinning: it means the
        # zero-batches trap comes from batch_size, not from the split.
        for total in range(1, 40):
            split = argmap.split_sizes(total, 0.5, 5)
            self.assertGreaterEqual(split.train, 1, f"total={total}")

    def test_dataset_too_small_to_train_is_refused(self):
        # Only reachable via a hand-edited profile: validation_split above 1.0
        # is outside the schema's bounds (which load() warns about).
        with self.assertRaises(argmap.ArgMapError) as ctx:
            argmap.check_dataset_math(5, 1.0, 5, 1)
        self.assertIn("nothing is left to train on", str(ctx.exception))

    def test_batch_size_above_train_split_is_refused_before_launch(self):
        with self.assertRaises(argmap.ArgMapError) as ctx:
            argmap.check_dataset_math(24, 0.1, 5, 40)
        message = str(ctx.exception)
        self.assertIn("zero batches", message)
        self.assertIn("17", message)  # 24 - 2 - 5

    def test_batch_size_equal_to_train_split_is_allowed(self):
        split = argmap.check_dataset_math(24, 0.1, 5, 17)
        self.assertEqual(split.train, 17)

    def test_build_reports_the_split(self):
        plan = argmap.build(make_profile(data__batch_size=8), total_utterances=100)
        self.assertIsNotNone(plan.split)
        self.assertEqual(plan.split.train, 85)
        self.assertTrue(any("85 train" in note for note in plan.notes), plan.notes)

    def test_build_refuses_an_oversized_batch(self):
        with self.assertRaises(argmap.ArgMapError):
            argmap.build(make_profile(data__batch_size=64), total_utterances=24)


class ClipLengthTest(unittest.TestCase):
    def test_short_clips_warn_about_silence_padding(self):
        plan = argmap.build(make_profile(), min_clip_seconds=0.3)
        self.assertTrue(
            any("zero-padded" in w for w in plan.warnings), plan.warnings
        )

    def test_one_second_clips_are_fine_at_default_segment_size(self):
        plan = argmap.build(make_profile(), min_clip_seconds=1.0)
        self.assertEqual([w for w in plan.warnings if "zero-padded" in w], [])

    def test_threshold_follows_segment_size_and_rate(self):
        # 8192/16000 = 0.512 s, so a 0.4 s clip is short at 16 kHz but not at 22.05 kHz
        warnings = argmap.check_clip_length(
            {"segment_size": 8192, "sample_rate": 16000}, 0.4
        )
        self.assertEqual(len(warnings), 1)
        self.assertEqual(
            argmap.check_clip_length(
                {"segment_size": 8192, "sample_rate": 22050}, 0.4
            ),
            [],
        )


class FinetuneTest(unittest.TestCase):
    MEDIUM_HPARAMS = {
        "sample_rate": 22050,
        "resblock": "2",
        "resblock_kernel_sizes": [3, 5, 7],
        "upsample_rates": [8, 8, 4],
        "upsample_initial_channel": 256,
        "upsample_kernel_sizes": [16, 16, 8],
        "num_symbols": 256,
        "num_speakers": 1,
    }

    def test_scratch_training_warns_that_finetuning_is_faster(self):
        plan = argmap.build(make_profile(finetune__mode="none"))
        self.assertTrue(
            any("from scratch" in w for w in plan.warnings), plan.warnings
        )
        self.assertIsNone(plan.ckpt_path)

    def test_ckpt_path_mode_puts_the_path_on_argv(self):
        plan = argmap.build(
            make_profile(finetune__mode="ckpt_path", finetune__checkpoint="/x/a.ckpt")
        )
        self.assertEqual(plan.ckpt_path, "/x/a.ckpt")
        self.assertIn("--ckpt_path", plan.argv)
        self.assertNotIn("warmstart_ckpt", plan.model)

    def test_warmstart_mode_uses_a_model_flag_not_ckpt_path(self):
        plan = argmap.build(
            make_profile(finetune__mode="warmstart", finetune__checkpoint="/x/a.ckpt")
        )
        self.assertIsNone(plan.ckpt_path)
        self.assertEqual(plan.model["warmstart_ckpt"], "/x/a.ckpt")

    def test_vocoder_warmstart_mode(self):
        plan = argmap.build(
            make_profile(
                finetune__mode="vocoder_warmstart", finetune__checkpoint="/x/a.ckpt"
            )
        )
        self.assertEqual(plan.model["vocoder_warmstart_ckpt"], "/x/a.ckpt")

    def test_missing_checkpoint_path_is_refused(self):
        with self.assertRaises(argmap.ArgMapError) as ctx:
            argmap.build(make_profile(finetune__mode="ckpt_path"))
        self.assertIn("finetune.checkpoint is empty", str(ctx.exception))

    def test_use_mrd_forbids_strict_ckpt_path(self):
        with self.assertRaises(argmap.ArgMapError) as ctx:
            argmap.build(
                make_profile(
                    model__use_mrd=True,
                    finetune__mode="ckpt_path",
                    finetune__checkpoint="/x/a.ckpt",
                )
            )
        message = str(ctx.exception)
        self.assertIn("strict load", message)
        self.assertIn("warmstart", message)

    def test_use_mrd_with_warmstart_is_allowed(self):
        plan = argmap.build(
            make_profile(
                model__use_mrd=True,
                finetune__mode="warmstart",
                finetune__checkpoint="/x/a.ckpt",
            )
        )
        self.assertTrue(plan.model["use_mrd"])

    def test_matching_checkpoint_architecture_passes(self):
        plan = argmap.build(
            make_profile(finetune__mode="ckpt_path", finetune__checkpoint="/x/a.ckpt"),
            ckpt_hparams=self.MEDIUM_HPARAMS,
        )
        self.assertEqual(plan.ckpt_path, "/x/a.ckpt")

    def test_high_profile_against_medium_checkpoint_is_refused(self):
        with self.assertRaises(argmap.ArgMapError) as ctx:
            argmap.build(
                make_profile(
                    voice__quality="high",
                    finetune__mode="ckpt_path",
                    finetune__checkpoint="/x/a.ckpt",
                ),
                ckpt_hparams=self.MEDIUM_HPARAMS,
            )
        message = str(ctx.exception)
        self.assertIn("does not match", message)
        self.assertIn("warmstart", message)

    def test_phoneme_count_mismatch_recommends_vocoder_warmstart(self):
        hparams = dict(self.MEDIUM_HPARAMS, num_symbols=512)
        with self.assertRaises(argmap.ArgMapError) as ctx:
            argmap.build(
                make_profile(
                    finetune__mode="ckpt_path", finetune__checkpoint="/x/a.ckpt"
                ),
                ckpt_hparams=hparams,
            )
        self.assertIn("vocoder_warmstart", str(ctx.exception))

    def test_tuple_and_list_hparams_compare_equal(self):
        # Checkpoints store tuples; presets use lists. They must not differ.
        arch = {
            **presets.MEDIUM,
            "sample_rate": 22050,
            "num_symbols": 256,
            "num_speakers": 1,
        }
        hparams = dict(self.MEDIUM_HPARAMS, upsample_rates=(8, 8, 4))
        self.assertEqual(argmap.compare_architecture(arch, hparams), [])

    def test_keys_absent_from_either_side_are_skipped(self):
        # num_symbols lives on the data side, so a model-only view lacks it.
        # Comparing it against None would reject every valid checkpoint.
        self.assertEqual(
            argmap.compare_architecture({"sample_rate": 22050}, self.MEDIUM_HPARAMS),
            [],
        )

    def test_warmstart_skips_the_architecture_check(self):
        # Non-strict loads tolerate mismatches; that is the whole point.
        plan = argmap.build(
            make_profile(
                voice__quality="high",
                finetune__mode="warmstart",
                finetune__checkpoint="/x/a.ckpt",
            ),
            ckpt_hparams=self.MEDIUM_HPARAMS,
        )
        self.assertEqual(plan.model["warmstart_ckpt"], "/x/a.ckpt")


class OfflineTest(unittest.TestCase):
    def test_offline_forces_mos_metric_off(self):
        plan = argmap.build(make_profile(), offline=True)
        self.assertEqual(plan.model["mos_metric"], "none")
        self.assertTrue(any("torch.hub" in n for n in plan.notes), plan.notes)

    def test_offline_note_says_it_is_an_optimisation_not_a_fix(self):
        # mos.py wraps torch.hub.load in try/except and sets _disabled, so
        # offline training works either way. The message must not overclaim.
        plan = argmap.build(make_profile(), offline=True)
        note = next(n for n in plan.notes if "torch.hub" in n)
        self.assertIn("not a crash fix", note)

    def test_offline_refuses_a_missing_local_checkpoint(self):
        with self.assertRaises(argmap.ArgMapError) as ctx:
            argmap.build(
                make_profile(
                    finetune__mode="ckpt_path",
                    finetune__checkpoint="/definitely/not/here.ckpt",
                ),
                offline=True,
            )
        self.assertIn("does not exist locally", str(ctx.exception))

    def test_mos_metric_none_is_respected_when_not_offline(self):
        plan = argmap.build(make_profile(model__mos_metric="none"))
        self.assertEqual(plan.model["mos_metric"], "none")


class CheckpointCallbackTest(unittest.TestCase):
    def test_changing_save_top_k_replicates_both_upstream_callbacks(self):
        plan = argmap.build(make_profile(trainer__checkpoint_save_top_k=2))
        callbacks = plan.trainer["callbacks"]
        self.assertEqual(len(callbacks), 2)
        monitors = [cb["init_args"]["monitor"] for cb in callbacks]
        self.assertEqual(monitors, ["val_mel", "val_mos"])
        self.assertEqual(
            [cb["init_args"]["save_top_k"] for cb in callbacks], [2, 2]
        )
        # Only the val_mel callback writes last.ckpt, matching upstream.
        self.assertEqual(
            [cb["init_args"]["save_last"] for cb in callbacks], [True, False]
        )

    def test_callback_filenames_match_upstream(self):
        callbacks = argmap.checkpoint_callbacks(5)
        self.assertEqual(
            callbacks[0]["init_args"]["filename"],
            "epoch={epoch}-val_mel={val_mel:.4f}",
        )
        self.assertFalse(callbacks[0]["init_args"]["auto_insert_metric_name"])

    def test_callbacks_survive_yaml_round_trip(self):
        from pipertrainer import yamlio

        plan = argmap.build(make_profile(trainer__checkpoint_save_top_k=3))
        text = yamlio.dumps(plan.config)
        restored = yamlio.loads(text)
        self.assertEqual(restored["trainer"]["callbacks"], plan.trainer["callbacks"])


class MiscTest(unittest.TestCase):
    def test_devices_digits_become_ints(self):
        plan = argmap.build(make_profile(trainer__devices="1"))
        self.assertEqual(plan.trainer["devices"], 1)

    def test_devices_auto_stays_a_string(self):
        self.assertEqual(argmap.build(make_profile()).trainer["devices"], "auto")

    def test_max_steps_only_emitted_when_positive(self):
        self.assertNotIn("max_steps", argmap.build(make_profile()).trainer)
        plan = argmap.build(make_profile(trainer__max_steps=100))
        self.assertEqual(plan.trainer["max_steps"], 100)

    def test_precision_change_warns(self):
        plan = argmap.build(make_profile(trainer__precision="16-mixed"))
        self.assertTrue(any("diverg" in w for w in plan.warnings), plan.warnings)

    def test_multispeaker_warns_that_the_pipeline_is_single_speaker(self):
        plan = argmap.build(make_profile(model__num_speakers=2))
        self.assertTrue(
            any("single-speaker" in w for w in plan.warnings), plan.warnings
        )

    def test_phonemes_path_only_emitted_when_set(self):
        plan = argmap.build(make_profile(data__phonemes_path="/x/p.json"))
        self.assertEqual(plan.data["phonemes_path"], "/x/p.json")

    def test_summarise_covers_the_decisions_that_matter(self):
        plan = argmap.build(make_profile(), total_utterances=100)
        labels = {label for label, _ in argmap.summarise(plan)}
        for expected in (
            "voice", "sample rate", "batch size", "precision", "split",
            "architecture", "starting from", "mos metric",
        ):
            self.assertIn(expected, labels)

    def test_low_vram_overlay_reports_what_it_changed(self):
        prof = make_profile()
        changed = argmap.apply_low_vram(prof)
        self.assertEqual(prof.data.batch_size, 8)
        self.assertTrue(any("data.batch_size" in c for c in changed), changed)

    def test_low_vram_overlay_is_idempotent(self):
        prof = make_profile()
        argmap.apply_low_vram(prof)
        self.assertEqual(argmap.apply_low_vram(prof), [])

    def test_oom_detection(self):
        self.assertTrue(argmap.looks_like_oom("torch.OutOfMemoryError: HIP out of memory"))
        self.assertTrue(argmap.looks_like_oom("CUDA error: out of memory"))
        self.assertFalse(argmap.looks_like_oom("loss_g=3.21"))

    def test_oom_ladder_is_ordered_cheapest_first(self):
        self.assertIn("batch_size", argmap.OOM_LADDER[0])
        self.assertIn("cpu", argmap.OOM_LADDER[-1])

    def test_cache_fingerprint_changes_with_metadata(self):
        plan = argmap.build(make_profile())
        first = argmap.cache_fingerprint_inputs(plan, b"1.wav|Hello.\n")
        second = argmap.cache_fingerprint_inputs(plan, b"1.wav|Hello there.\n")
        self.assertNotEqual(first, second)

    def test_cache_fingerprint_changes_with_caching_settings(self):
        base = argmap.build(make_profile())
        changed = argmap.build(make_profile(data__trim_silence=False))
        payload = b"1.wav|Hello.\n"
        self.assertNotEqual(
            argmap.cache_fingerprint_inputs(base, payload),
            argmap.cache_fingerprint_inputs(changed, payload),
        )

    def test_cache_fingerprint_ignores_unrelated_settings(self):
        # batch size does not affect the cached tensors, so it must not
        # invalidate hours of preprocessing.
        base = argmap.build(make_profile(data__batch_size=8))
        other = argmap.build(make_profile(data__batch_size=16))
        payload = b"1.wav|Hello.\n"
        self.assertEqual(
            argmap.cache_fingerprint_inputs(base, payload),
            argmap.cache_fingerprint_inputs(other, payload),
        )


if __name__ == "__main__":
    unittest.main()
