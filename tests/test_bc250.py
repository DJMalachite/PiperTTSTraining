"""The BC-250 build path: pins, named environments, and the arch-list check.

None of this needs a GPU, a venv, or Linux — which is the point. The parts of
the gfx1013 story that can be decided by pure logic are decided here, so that
what is left to measure on the board itself is only what genuinely has to be.
"""

from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import _support  # noqa: F401

from pipertrainer import env as env_mod
from pipertrainer import hardware as H
from pipertrainer import install
from pipertrainer import paths as paths_mod
from pipertrainer import pins as pins_mod


class BC250PinsTest(unittest.TestCase):
    """The scripts read these too, so a typo here is a broken build there."""

    def setUp(self):
        self.pins = pins_mod.load().bc250

    def test_reference_repos_are_pinned_to_full_shas(self):
        for sha in (self.pins.reference_sha, self.pins.cu_unlock_sha):
            with self.subTest(sha=sha):
                self.assertEqual(len(sha), 40)
                self.assertTrue(all(c in "0123456789abcdef" for c in sha))

    def test_both_kernel_side_projects_are_recorded(self):
        # The 40-CU unlock and the flush fixes come from different people; the
        # attribution has to survive in the pins, not just in the docs.
        self.assertIn("akandr", self.pins.reference_repo)
        self.assertIn("duggasco", self.pins.cu_unlock_repo)

    def test_torch_version_matches_the_rocm_pin(self):
        # The from-source build deliberately carries the same base version as
        # the wheel it replaces, so PIP_CONSTRAINT and every version check
        # behave identically whichever one is installed.
        rocm = pins_mod.load().torch("rocm")
        self.assertEqual(self.pins.torch_tag.lstrip("v"), rocm.version)
        self.assertEqual(
            self.pins.torch_version, f"{rocm.version}+{self.pins.torch_local_version}"
        )

    def test_container_python_is_not_the_newest(self):
        # numba, which openai-whisper needs, lags new CPython. The container has
        # to agree with [python].prefer or install._pick_python will disagree
        # with the packages stage-03 installed.
        self.assertEqual(self.pins.container_python, pins_mod.load().python_prefer[0])

    def test_build_free_gib_is_an_int(self):
        self.assertIsInstance(self.pins.build_free_gib, int)
        self.assertGreater(self.pins.build_free_gib, 20)

    def test_a_missing_section_says_so(self):
        empty = pins_mod.Pins(raw={})
        with self.assertRaises(pins_mod.PinsError):
            empty.bc250

    def test_an_incomplete_section_names_what_is_missing(self):
        partial = pins_mod.Pins(raw={"bc250": {"gfx_target": "gfx1013"}})
        with self.assertRaises(pins_mod.PinsError) as caught:
            partial.bc250
        self.assertIn("reference_sha", str(caught.exception))


class NamedEnvironmentTest(unittest.TestCase):
    """PIPERTRAINER_ENV, which lets host and container coexist in one clone."""

    def suffix(self, value):
        with mock.patch.dict(os.environ, {"PIPERTRAINER_ENV": value}, clear=False):
            return paths_mod.env_suffix()

    def test_unset_is_the_historic_layout(self):
        environ = dict(os.environ)
        environ.pop("PIPERTRAINER_ENV", None)
        with mock.patch.dict(os.environ, environ, clear=True):
            self.assertEqual(paths_mod.env_suffix(), "")

    def test_a_name_becomes_a_suffix(self):
        self.assertEqual(self.suffix("bc250"), "-bc250")

    def test_whitespace_is_not_a_name(self):
        self.assertEqual(self.suffix("   "), "")

    def test_a_name_cannot_escape_the_repo(self):
        # It ends up in a path, so it goes through the same slug the voice names
        # do rather than being trusted.
        self.assertEqual(self.suffix("../../etc"), "-etc")
        self.assertNotIn("/", self.suffix("a/b"))

    def test_names_the_run_shim_accepts_are_left_alone(self):
        # ./run refuses anything it would have to sanitise, precisely so that
        # the shell and paths.py always derive the same directory. Any name
        # matching that rule must therefore pass through slug unchanged.
        for name in ("bc250", "a.b_c-1", "x", "cuda2"):
            with self.subTest(name=name):
                self.assertEqual(self.suffix(name), f"-{name}")

    def test_the_constants_actually_follow_it(self):
        with mock.patch.dict(os.environ, {"PIPERTRAINER_ENV": "bc250"}, clear=False):
            reloaded = importlib.reload(paths_mod)
            try:
                self.assertEqual(reloaded.VENV_DIR.name, ".venv-bc250")
                self.assertEqual(reloaded.STATE_DIR.name, ".state-bc250")
                self.assertEqual(reloaded.VENV_DIR.parent, reloaded.REPO_ROOT)
            finally:
                environ = dict(os.environ)
                environ.pop("PIPERTRAINER_ENV", None)
                with mock.patch.dict(os.environ, environ, clear=True):
                    importlib.reload(paths_mod)
        self.assertEqual(paths_mod.VENV_DIR.name, ".venv")
        self.assertEqual(paths_mod.STATE_DIR.name, ".state")


class CompiledForDeviceTest(unittest.TestCase):
    """The check that names the real cause of 'invalid device function'.

    The BC-250's signature is a device that passes every availability check and
    every matmul, and has no kernels: GEMM comes from rocBLAS, which has its own
    target list. So this question is asked independently of whether matmul
    worked, and it must never guess.
    """

    def test_a_listed_target_is_compiled(self):
        self.assertIs(
            env_mod.compiled_for_device("gfx1030", ["gfx906", "gfx1030"]), True
        )

    def test_a_missing_target_is_not(self):
        self.assertIs(
            env_mod.compiled_for_device("gfx1013", ["gfx1030", "gfx1100"]), False
        )

    def test_feature_suffixes_do_not_confuse_it(self):
        self.assertIs(
            env_mod.compiled_for_device(
                "gfx1013:sramecc-:xnack-", ["gfx1013:xnack-", "gfx1030"]
            ),
            True,
        )

    def test_an_empty_arch_list_is_unknowable(self):
        # A false "no" would send someone off to rebuild torch for no reason.
        self.assertIsNone(env_mod.compiled_for_device("gfx1013", []))

    def test_no_reported_target_is_unknowable(self):
        for arch in (None, "", "   "):
            with self.subTest(arch=arch):
                self.assertIsNone(env_mod.compiled_for_device(arch, ["gfx1030"]))

    def test_cuda_is_never_answered(self):
        # sm_86 and gfx1013 are not comparable; a CUDA build must not be told
        # its kernels are missing.
        self.assertIsNone(env_mod.compiled_for_device("8.6", ["sm_86", "sm_90"]))

    def test_the_torchinfo_property_agrees(self):
        info = env_mod.TorchInfo(
            ok=True, gcn_arch="gfx1013", arch_list=["gfx1030", "gfx1100"]
        )
        self.assertIs(info.compiled_for_device, False)
        self.assertIs(
            env_mod.TorchInfo(
                ok=True, gcn_arch="gfx1013", arch_list=["gfx1013"]
            ).compiled_for_device,
            True,
        )

    def test_matmul_working_does_not_imply_compiled(self):
        # The whole reason this check exists.
        info = env_mod.TorchInfo(
            ok=True,
            available=True,
            matmul_ok=True,
            gcn_arch="gfx1013",
            arch_list=["gfx1030"],
        )
        self.assertTrue(info.usable_gpu)
        self.assertIs(info.compiled_for_device, False)


class LocalWheelTest(unittest.TestCase):
    """--torch-spec pointing at a wheel that no index has ever heard of."""

    def test_a_requirement_is_not_a_wheel(self):
        for spec in ("torch==2.9.1", "torch", "torch>=2,<3", ""):
            with self.subTest(spec=spec):
                self.assertIsNone(install.local_wheel(spec))

    def test_a_missing_file_is_not_a_wheel(self):
        self.assertIsNone(install.local_wheel("/nowhere/torch-2.9.1.whl"))

    def test_an_existing_wheel_is_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "torch-2.9.1+rocm6.4.gfx1013-cp313-linux_x86_64.whl"
            wheel.write_bytes(b"")
            found = install.local_wheel(str(wheel))
            self.assertIsNotNone(found)
            self.assertEqual(found.name, wheel.name)

    def test_a_directory_named_whl_is_not_a_wheel(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "torch.whl"
            fake.mkdir()
            self.assertIsNone(install.local_wheel(str(fake)))


class BuildScriptAdviceTest(unittest.TestCase):
    """A blocked board should be told what to build, not just what not to try."""

    def test_bc250_is_pointed_at_the_build_script(self):
        advice = H.unsupported_arch_advice(H.BC250)
        self.assertIn("scripts/bc250/build.sh", advice)

    def test_it_still_refuses_to_suggest_swapping_wheels(self):
        # The correction this repo already made must survive the addition.
        advice = H.unsupported_arch_advice(H.BC250)
        for wrong in ("--torch-index", "--torch-spec", "rocm6.3"):
            self.assertNotIn(wrong, advice)

    def test_generic_hardware_is_unaffected(self):
        self.assertEqual(H.unsupported_arch_advice(H.GENERIC), H.GENERIC_ARCH_ADVICE)

    def test_the_script_is_where_the_profile_says_it_is(self):
        self.assertTrue((_support.REPO_ROOT / H.BC250.build_script).is_file())


class BC250EnvironmentTest(unittest.TestCase):
    """Every variable in the profile needs a reason, so assert the reasons."""

    def test_sdma_is_still_disabled(self):
        self.assertEqual(H.BC250.env["HSA_ENABLE_SDMA"], "0")

    def test_hipblaslt_is_refused(self):
        # hipBLASLt has no gfx1013 support of any kind, and torch reaches for it
        # first; without this the built rocBLAS never gets used.
        self.assertEqual(H.BC250.env["TORCH_BLAS_PREFER_HIPBLASLT"], "0")

    def test_miopen_does_not_search_exhaustively(self):
        self.assertEqual(H.BC250.env["MIOPEN_FIND_MODE"], "FAST")

    def test_the_banned_override_is_not_quietly_reintroduced(self):
        for key in H.BC250.env:
            self.assertNotIn(key, H.BC250.banned_env)


if __name__ == "__main__":
    unittest.main()
