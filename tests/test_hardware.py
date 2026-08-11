"""Hardware profiles and the gfx override table.

The BC-250 entries encode findings from https://github.com/akandr/bc250-rocm.
The one that matters most is negative: `HSA_OVERRIDE_GFX_VERSION` must never be
suggested for gfx1013, because the aperture layout differs and anything using
scratch addressing then raises HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION. An
earlier version of this repo applied it automatically, so there is a test.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from . import _support  # noqa: F401

from pipertrainer import env as env_mod
from pipertrainer import hardware as H
from pipertrainer import profile as P


class DetectionTest(unittest.TestCase):
    def test_gfx1013_selects_the_bc250_profile(self):
        self.assertIs(H.detect("gfx1013"), H.BC250)

    def test_gfx1013_with_a_feature_suffix_still_matches(self):
        # gcnArchName often looks like "gfx1013:sramecc-:xnack-".
        self.assertIs(H.detect("gfx1013:sramecc-:xnack-"), H.BC250)

    def test_other_targets_are_generic(self):
        for arch in ("gfx1030", "gfx1100", "gfx906", "", None):
            with self.subTest(arch=arch):
                self.assertIs(H.detect(arch), H.GENERIC)

    def test_explicit_choice_beats_detection(self):
        self.assertIs(H.resolve("generic", "gfx1013"), H.GENERIC)
        self.assertIs(H.resolve("bc250", "gfx1030"), H.BC250)

    def test_auto_falls_back_to_detection(self):
        self.assertIs(H.resolve("auto", "gfx1013"), H.BC250)
        self.assertIs(H.resolve("", "gfx1013"), H.BC250)

    def test_unknown_profile_name_raises(self):
        with self.assertRaises(ValueError):
            H.get("bc300")


class OverrideTest(unittest.TestCase):
    """The correction: gfx1013 must never get an override suggestion."""

    def info(self, arch, arch_list=("gfx1030",)):
        return env_mod.TorchInfo(
            ok=True,
            hip="6.4.0",
            available=True,
            gcn_arch=arch,
            arch_list=list(arch_list),
        )

    def test_no_override_is_suggested_for_gfx1013(self):
        self.assertIsNone(env_mod.needs_gfx_override(self.info("gfx1013")))

    def test_gfx1013_is_absent_from_the_override_table(self):
        self.assertNotIn("gfx1013", env_mod.GFX_OVERRIDES)

    def test_rdna2_variants_map_to_gfx1030(self):
        for arch in ("gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036"):
            with self.subTest(arch=arch):
                self.assertEqual(env_mod.needs_gfx_override(self.info(arch)), "10.3.0")

    def test_rdna1_variants_map_to_gfx1010_not_gfx1030(self):
        # An earlier version mapped these to 10.3.0, which is the wrong family.
        for arch in ("gfx1011", "gfx1012"):
            with self.subTest(arch=arch):
                self.assertEqual(env_mod.needs_gfx_override(self.info(arch)), "10.1.0")

    def test_unlisted_rdna2_target_gets_a_blind_gfx1030_suggestion(self):
        self.assertEqual(
            env_mod.needs_gfx_override(self.info("gfx1038", ("gfx1030",))), "10.3.0"
        )

    def test_unlisted_non_rdna2_target_gets_no_guess(self):
        self.assertIsNone(
            env_mod.needs_gfx_override(self.info("gfx90c", ("gfx1030",)))
        )

    def test_supported_target_needs_no_override(self):
        self.assertIsNone(
            env_mod.needs_gfx_override(self.info("gfx1030", ("gfx1030",)))
        )

    def test_cuda_never_gets_an_override(self):
        cuda = env_mod.TorchInfo(ok=True, cuda="12.8", available=True)
        self.assertIsNone(env_mod.needs_gfx_override(cuda))


class BC250ProfileTest(unittest.TestCase):
    def test_sdma_is_disabled(self):
        # The SDMA host-to-device path is broken for bulk transfers.
        self.assertEqual(H.BC250.env.get("HSA_ENABLE_SDMA"), "0")

    def test_the_override_is_banned_with_a_reason(self):
        self.assertIn("HSA_OVERRIDE_GFX_VERSION", H.BC250.banned_env)
        reason = H.BC250.banned_env["HSA_OVERRIDE_GFX_VERSION"]
        self.assertIn("aperture", reason.lower())

    def test_training_is_marked_blocked(self):
        self.assertEqual(H.BC250.training, H.BLOCKED)
        self.assertEqual(H.GENERIC.training, H.SUPPORTED)

    def test_caveats_mention_the_actual_blockers(self):
        text = " ".join(H.BC250.caveats).lower()
        self.assertIn("elementwise", text)
        self.assertIn("invalid device function", text)
        self.assertIn("allocate and free", text)

    def test_it_cites_its_source(self):
        self.assertTrue(H.BC250.reference.startswith("https://"))

    def test_forced_settings_are_conservative(self):
        self.assertEqual(H.BC250.settings["trainer.precision"], "32-true")
        self.assertLessEqual(H.BC250.settings["data.batch_size"], 8)
        self.assertLessEqual(H.BC250.settings["data.num_workers"], 2)


class UnsupportedArchAdviceTest(unittest.TestCase):
    """The index-swap suggestion is a dead end on a BLOCKED board.

    No stock ROCm wheel of any version ships gfx1013 Tensile kernels, so
    telling a BC-250 owner to re-download torch from another index costs
    gigabytes and fails identically. Same class of mistake as suggesting
    HSA_OVERRIDE_GFX_VERSION.
    """

    def test_generic_hardware_gets_the_index_swap(self):
        advice = H.unsupported_arch_advice(H.GENERIC)
        self.assertIn("--torch-index", advice)
        self.assertIn("pins.toml", advice)

    def test_an_unlisted_card_still_gets_the_index_swap(self):
        # gfx1031 is a normal RDNA2 laptop part missing from some builds:
        # a different wheel genuinely can fix it.
        advice = H.unsupported_arch_advice(H.detect("gfx1031"))
        self.assertIn("--torch-index", advice)

    def test_bc250_is_never_told_to_swap_the_torch_index(self):
        advice = H.unsupported_arch_advice(H.detect("gfx1013:xnack-"))
        self.assertNotIn("--torch-index", advice)
        self.assertNotIn("--torch-spec", advice)
        self.assertNotIn("rocm6.3", advice)

    def test_bc250_is_pointed_at_the_documentation_instead(self):
        advice = H.unsupported_arch_advice(H.detect("gfx1013"))
        self.assertIn("docs/BC250.md", advice)
        self.assertIn(H.BC250_REFERENCE, advice)

    def test_every_blocked_profile_documents_itself(self):
        # The advice is only useful because it has somewhere to send people.
        for profile in H.PROFILES.values():
            if profile.training == H.BLOCKED:
                with self.subTest(profile=profile.name):
                    self.assertTrue(profile.doc or profile.reference)


class ApplyTest(unittest.TestCase):
    def test_apply_forces_settings_and_reports_them(self):
        prof = P.Profile()
        prof.data.batch_size = 32
        changed = H.apply(H.BC250, prof)
        self.assertEqual(prof.data.batch_size, 4)
        self.assertEqual(prof.trainer.precision, "32-true")
        self.assertTrue(any("data.batch_size" in line for line in changed))

    def test_apply_sets_the_environment(self):
        prof = P.Profile()
        H.apply(H.BC250, prof)
        self.assertEqual(prof.runtime.env["HSA_ENABLE_SDMA"], "0")

    def test_apply_removes_a_banned_variable(self):
        prof = P.Profile()
        prof.runtime.env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
        changed = H.apply(H.BC250, prof)
        self.assertNotIn("HSA_OVERRIDE_GFX_VERSION", prof.runtime.env)
        self.assertTrue(any("removed" in line for line in changed))

    def test_apply_is_idempotent(self):
        prof = P.Profile()
        H.apply(H.BC250, prof)
        self.assertEqual(H.apply(H.BC250, prof), [])

    def test_generic_changes_nothing(self):
        prof = P.Profile()
        before = P.to_dict(prof)
        self.assertEqual(H.apply(H.GENERIC, prof), [])
        self.assertEqual(P.to_dict(prof), before)


class SchemaTest(unittest.TestCase):
    def test_profile_choices_match_the_hardware_module(self):
        # profile.py duplicates the names as a literal to avoid importing the
        # probing machinery; this asserts the duplication stays honest.
        self.assertEqual(tuple(P.HARDWARE_NAMES), H.NAMES)

    def test_default_is_auto(self):
        self.assertEqual(P.Profile().runtime.hardware, "auto")

    def test_hardware_setting_round_trips(self):
        prof = P.Profile()
        prof.runtime.hardware = "bc250"
        restored, warnings = P.from_dict(P.to_dict(prof))
        self.assertEqual(warnings, [])
        self.assertEqual(restored.runtime.hardware, "bc250")


class KernelParsingTest(unittest.TestCase):
    def test_version_tuple(self):
        self.assertEqual(H.kernel_version_tuple("7.1.5-arch1-1"), (7, 1, 5))
        self.assertEqual(H.kernel_version_tuple("6.18.0"), (6, 18, 0))
        self.assertEqual(H.kernel_version_tuple("7.2"), (7, 2))
        self.assertEqual(H.kernel_version_tuple("nonsense"), ())

    def test_minimum_kernel_ordering(self):
        self.assertLess(H.kernel_version_tuple("6.18.0"), H.MIN_BC250_KERNEL)
        self.assertGreaterEqual(H.kernel_version_tuple("7.1.5"), H.MIN_BC250_KERNEL)
        self.assertGreaterEqual(H.kernel_version_tuple("7.2.0"), H.MIN_BC250_KERNEL)

    def test_cmdline_param_extraction(self):
        cmdline = (
            "root=/dev/sda2 amdgpu.bc250_cc_write_mode=3 "
            "amdgpu.bc250_flush_by_runlist=1 quiet"
        )
        self.assertEqual(H.cmdline_param(cmdline, "amdgpu.bc250_cc_write_mode"), "3")
        self.assertEqual(H.cmdline_param(cmdline, "amdgpu.bc250_flush_by_runlist"), "1")
        self.assertIsNone(H.cmdline_param(cmdline, "amdgpu.sched_policy"))
        self.assertEqual(H.cmdline_param(cmdline, "quiet"), "")

    def test_expected_module_parameters(self):
        self.assertEqual(H.BC250_MODPARAMS["bc250_cc_write_mode"], "3")
        self.assertEqual(H.BC250_MODPARAMS["bc250_flush_by_runlist"], "1")


class RocblasDetectionTest(unittest.TestCase):
    def test_missing_library_directory_is_unknown_not_false(self):
        from pathlib import Path

        self.assertIsNone(H.rocblas_has_gfx1013([Path("/nonexistent/torch/lib")]))

    def test_present_and_absent_are_distinguished(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            library = Path(tmp) / "rocblas" / "library"
            library.mkdir(parents=True)
            (library / "Kernels.so-000-gfx1030.hsaco").write_bytes(b"")
            self.assertIs(H.rocblas_has_gfx1013([Path(tmp)]), False)
            (library / "Kernels.so-000-gfx1013.hsaco").write_bytes(b"")
            self.assertIs(H.rocblas_has_gfx1013([Path(tmp)]), True)

    def test_the_system_rocm_is_searched_too(self):
        # A torch built from source links against the system rocBLAS instead of
        # bundling one, so looking only inside the wheel would make the answer
        # depend on how torch was installed rather than on what is on disk.
        dirs = [str(p) for p in H.rocblas_library_dirs()]
        for prefix in ("/opt/rocm/lib", "/opt/bc250/rocm/lib"):
            with self.subTest(prefix=prefix):
                self.assertIn(f"{prefix}/rocblas/library".replace("/", os.sep), dirs)

    def test_the_tensile_override_is_honoured_and_comes_first(self):
        # ROCBLAS_TENSILE_LIBPATH is rocBLAS's own variable and points straight
        # at the library directory, so it is not suffixed like the others.
        with mock.patch.dict(
            os.environ, {"ROCBLAS_TENSILE_LIBPATH": "/somewhere/library"}
        ):
            dirs = H.rocblas_library_dirs()
        self.assertEqual(str(dirs[0]), os.path.normpath("/somewhere/library"))

    def test_the_search_order_has_no_duplicates(self):
        from pathlib import Path

        dirs = H.rocblas_library_dirs([Path("/opt/rocm/lib")])
        self.assertEqual(len(dirs), len(set(str(d) for d in dirs)))


class TrainingProbeTest(unittest.TestCase):
    """The verdict text has to explain, not just report a code."""

    def test_missing_kernels_are_explained(self):
        probe = env_mod.TrainingProbe(
            ok=False,
            iterations=0,
            error="RuntimeError: invalid device function",
        )
        self.assertIn("missing GPU kernels", probe.verdict)
        self.assertIn("blocks training", probe.verdict)

    def test_partial_progress_is_reported(self):
        probe = env_mod.TrainingProbe(
            ok=False, iterations=27, requested=60, error="memory access fault"
        )
        self.assertIn("27 of 60", probe.verdict)

    def test_abort_is_distinguished_from_an_exception(self):
        probe = env_mod.TrainingProbe(ok=False, aborted=True, iterations=3)
        self.assertIn("aborted", probe.verdict)

    def test_success_reports_the_count(self):
        probe = env_mod.TrainingProbe(ok=True, iterations=60, requested=60)
        self.assertIn("60 autograd iterations", probe.verdict)


if __name__ == "__main__":
    unittest.main()
