"""Cross-platform behaviour: layout, entry points, and persisted environment.

This tool runs on Linux and on Windows, and the maintainer can only ever be
sitting at one of them. Everything here is therefore a fact that can be checked
without the other platform present: the venv layout is derived, not probed; the
entry-point scripts are inspected as bytes; the GPU-override table is pure data.

The line-ending tests look pedantic and are not. cmd.exe parses a batch file
with LF-only line endings by eating the first character of every line, so `rem`
runs as `m` and the failure looks nothing like a newline problem. A `.sh` file
with CRLF fails just as opaquely, with `bad interpreter: /bin/sh^M`.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import _support

from pipertrainer import env as env_mod
from pipertrainer import paths, proc
from pipertrainer import profile as profile_mod


class VenvLayoutTest(unittest.TestCase):
    """Windows puts venv executables in Scripts/ with an .exe suffix."""

    def test_the_layout_matches_the_running_platform(self):
        if sys.platform == "win32":
            self.assertEqual(paths.VENV_BINDIR, "Scripts")
            self.assertEqual(paths.EXE_SUFFIX, ".exe")
        else:
            self.assertEqual(paths.VENV_BINDIR, "bin")
            self.assertEqual(paths.EXE_SUFFIX, "")

    def test_venv_python_lives_in_the_platform_bindir(self):
        python = paths.venv_python()
        self.assertEqual(python.parent.name, paths.VENV_BINDIR)
        self.assertEqual(python.name, "python" + paths.EXE_SUFFIX)

    def test_every_venv_executable_goes_through_one_helper(self):
        # tensorboard is a console script, so it has the same suffix problem as
        # the interpreter and must not be spelled by hand anywhere.
        self.assertEqual(
            paths.venv_bin("tensorboard").name, "tensorboard" + paths.EXE_SUFFIX
        )

    def test_the_bindir_is_the_parent_of_the_executables(self):
        self.assertEqual(paths.venv_bindir(), paths.venv_python().parent)


class EnvNameTest(unittest.TestCase):
    """PIPERTRAINER_ENV is refused rather than sanitised.

    ``run.cmd`` cannot express the rule cheaply in batch, so it defers to this
    check. Both entry points therefore have to agree with it exactly.
    """

    def problem(self, value):
        with mock.patch.dict(os.environ, {"PIPERTRAINER_ENV": value}, clear=False):
            return paths.env_name_problem()

    def test_unset_is_fine(self):
        environ = dict(os.environ)
        environ.pop("PIPERTRAINER_ENV", None)
        with mock.patch.dict(os.environ, environ, clear=True):
            self.assertIsNone(paths.env_name_problem())

    def test_ordinary_names_are_accepted(self):
        for name in ("cuda", "wsl", "a.b_c-1", "x", "cuda2"):
            self.assertIsNone(self.problem(name), name)

    def test_names_that_slug_would_rewrite_are_refused(self):
        # If slug() would change the name, the entry-point script and paths.py
        # would look in different directories. Refusing keeps them in step.
        for name in ("bad name", "has/slash", "-leading", "trailing-", "!"):
            self.assertIsNotNone(self.problem(name), name)

    def test_the_message_names_the_variable_and_the_value(self):
        message = self.problem("bad name")
        self.assertIn("PIPERTRAINER_ENV", message)
        self.assertIn("bad name", message)

    def test_an_accepted_name_survives_slug_unchanged(self):
        for name in ("cuda", "a.b_c-1"):
            self.assertEqual(paths.slug(name), name)


class PersistedEnvTest(unittest.TestCase):
    """.state/env.json replaces the shell fragment ./run used to source."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        state = Path(self.tmp.name)
        self.patches = [
            mock.patch.object(env_mod, "ENV_JSON", state / "env.json"),
            mock.patch.object(env_mod, "STATE_DIR", state),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_missing_file_reads_as_empty(self):
        self.assertEqual(env_mod.read_persisted_env(), {})

    def test_round_trip(self):
        env_mod.write_persisted_env({"HSA_OVERRIDE_GFX_VERSION": "10.3.0"})
        self.assertEqual(
            env_mod.read_persisted_env(), {"HSA_OVERRIDE_GFX_VERSION": "10.3.0"}
        )

    def test_writing_merges_rather_than_replaces(self):
        env_mod.write_persisted_env({"A": "1"})
        env_mod.write_persisted_env({"B": "2"})
        self.assertEqual(env_mod.read_persisted_env(), {"A": "1", "B": "2"})

    def test_the_embedded_comment_is_not_returned_as_a_variable(self):
        env_mod.write_persisted_env({"A": "1"})
        self.assertNotIn("_comment", env_mod.read_persisted_env())

    def test_corrupt_json_reads_as_empty_rather_than_raising(self):
        # A half-written state file must not stop the tool from starting; the
        # worst case is re-detecting what it describes.
        env_mod.ENV_JSON.parent.mkdir(parents=True, exist_ok=True)
        env_mod.ENV_JSON.write_text("{not json", encoding="utf-8")
        self.assertEqual(env_mod.read_persisted_env(), {})

    def test_applying_does_not_clobber_the_shell(self):
        # The file records what setup *discovered*. An explicit value in the
        # environment is the user saying they know better.
        env_mod.write_persisted_env({"HSA_OVERRIDE_GFX_VERSION": "10.3.0"})
        with mock.patch.dict(
            os.environ, {"HSA_OVERRIDE_GFX_VERSION": "11.0.0"}, clear=False
        ):
            env_mod.apply_persisted_env()
            self.assertEqual(os.environ["HSA_OVERRIDE_GFX_VERSION"], "11.0.0")

    def test_applying_sets_what_the_shell_did_not(self):
        env_mod.write_persisted_env({"PT_TEST_ONLY": "yes"})
        environ = dict(os.environ)
        environ.pop("PT_TEST_ONLY", None)
        with mock.patch.dict(os.environ, environ, clear=True):
            env_mod.apply_persisted_env()
            self.assertEqual(os.environ["PT_TEST_ONLY"], "yes")


class CommandQuotingTest(unittest.TestCase):
    """A recorded command has to be one the user can paste back."""

    def test_quoting_matches_the_platform_shell(self):
        rendered = proc.describe(["python", "-c", "import sys"])
        if sys.platform == "win32":
            # cmd.exe does not understand POSIX single quotes.
            self.assertNotIn("'", rendered)
            self.assertIn('"import sys"', rendered)
        else:
            self.assertIn("'import sys'", rendered)

    def test_paths_with_spaces_survive(self):
        rendered = proc.describe([Path("/opt/some dir/python")])
        self.assertIn("some dir", rendered)
        # Quoted somehow — the point is that it is not bare.
        self.assertNotEqual(rendered, "/opt/some dir/python")


class EntryPointTest(unittest.TestCase):
    """The four entry-point scripts, checked as bytes.

    Line endings are load-bearing here in opposite directions, and neither
    failure names newlines when it happens.
    """

    def read(self, name):
        return (_support.REPO_ROOT / name).read_bytes()

    def test_the_posix_entry_points_exist_and_are_lf(self):
        for name in ("run", "setup", "scripts/smoke_test.sh"):
            data = self.read(name)
            self.assertTrue(data.startswith(b"#!"), f"{name} lost its shebang")
            self.assertNotIn(b"\r\n", data, f"{name} must be LF: CRLF breaks sh")

    def test_the_windows_entry_points_exist_and_are_crlf(self):
        for name in ("run.cmd", "setup.cmd", "scripts/smoke_test.cmd"):
            data = self.read(name)
            self.assertNotIn(
                b"\n", data.replace(b"\r\n", b""),
                f"{name} must be CRLF: cmd.exe eats the first character of "
                f"every LF-terminated line",
            )

    def test_both_platforms_can_reach_every_command(self):
        posix = self.read("run").decode("utf-8")
        windows = self.read("run.cmd").decode("utf-8")
        for script in (posix, windows):
            self.assertIn("pipertrainer", script)
            self.assertIn("PYTHONPATH", script)
            self.assertIn("PIPERTRAINER_ENV", script)

    def test_neither_entry_point_sources_the_old_shell_fragment(self):
        # env.sh was replaced by env.json precisely because run.cmd could not
        # source it. A reintroduced `. env.sh` would work on one platform only.
        self.assertNotIn("env.sh", self.read("run").decode("utf-8"))

    def test_the_windows_shim_looks_in_the_windows_venv(self):
        self.assertIn("Scripts\\python.exe", self.read("run.cmd").decode("utf-8"))


class GfxOverrideTest(unittest.TestCase):
    """HSA_OVERRIDE_GFX_VERSION is applied only where it is known to help.

    An override makes a device claim an ISA it does not have. Where that is
    wrong it converts a clear "no kernels for this architecture" into a fault
    inside the runtime, so the table is an allowlist and stays one.
    """

    def info(self, arch, arch_list=()):
        return env_mod.TorchInfo(
            ok=True,
            hip="6.4",
            available=True,
            gcn_arch=arch,
            arch_list=list(arch_list),
        )

    def test_known_targets_are_rescued(self):
        self.assertEqual(env_mod.needs_gfx_override(self.info("gfx1012")), "10.1.0")
        self.assertEqual(env_mod.needs_gfx_override(self.info("gfx1032")), "10.3.0")

    def test_a_cuda_build_is_never_given_a_rocm_override(self):
        cuda = env_mod.TorchInfo(ok=True, cuda="12.8", available=True, gcn_arch="sm_86")
        self.assertIsNone(env_mod.needs_gfx_override(cuda))

    def test_no_override_without_a_visible_device(self):
        self.assertIsNone(
            env_mod.needs_gfx_override(
                env_mod.TorchInfo(ok=True, hip="6.4", available=False)
            )
        )

    def test_an_unlisted_target_outside_rdna2_is_not_guessed(self):
        # Guessing here would trade a legible failure for a silent one.
        self.assertIsNone(
            env_mod.needs_gfx_override(self.info("gfx900", ["gfx1030"]))
        )

    def test_every_mapping_is_a_version_triple(self):
        for target, version in env_mod.GFX_OVERRIDES.items():
            self.assertTrue(target.startswith("gfx"), target)
            self.assertEqual(len(version.split(".")), 3, version)


class CompiledForDeviceTest(unittest.TestCase):
    """The check that predicts training independently of whether matmul works."""

    def test_a_device_missing_from_the_arch_list_is_reported(self):
        self.assertIs(
            env_mod.compiled_for_device("gfx1010", ["gfx1030", "gfx1100"]), False
        )

    def test_feature_suffixes_do_not_defeat_the_match(self):
        # gcnArchName is usually "gfx1030:sramecc-:xnack-".
        self.assertIs(
            env_mod.compiled_for_device(
                "gfx1030:sramecc-:xnack-", ["gfx1030:xnack-", "gfx1100"]
            ),
            True,
        )

    def test_an_empty_arch_list_is_unanswerable_rather_than_no(self):
        # A false 'no' would send someone off to rebuild torch for no reason.
        self.assertIsNone(env_mod.compiled_for_device("gfx1030", []))
        self.assertIsNone(env_mod.compiled_for_device(None, ["gfx1030"]))

    def test_a_cuda_target_is_unanswerable(self):
        self.assertIsNone(env_mod.compiled_for_device("sm_86", ["sm_86"]))


class ArchAdviceTest(unittest.TestCase):
    """What to try differs by platform, because the delivery differs."""

    def test_windows_is_not_sent_looking_for_another_index(self):
        # AMD publishes exactly one ROCm build for native Windows, at fixed
        # URLs. Suggesting an index swap would send someone after wheels that
        # do not exist.
        with mock.patch.object(env_mod, "WINDOWS", True):
            advice = env_mod.unsupported_arch_advice()
        self.assertNotIn("--torch-index", advice)
        self.assertIn("driver", advice)

    def test_linux_is_pointed_at_another_index(self):
        with mock.patch.object(env_mod, "WINDOWS", False):
            advice = env_mod.unsupported_arch_advice()
        self.assertIn("--torch-index", advice)


class WindowsRocmPinTest(unittest.TestCase):
    """AMD ships ROCm for native Windows as wheel URLs, not a pip index.

    The constraints are real and all three break the install if ignored: the
    wheels are cp312-only, the ROCm runtime is a separate set that must land
    first, and download.pytorch.org has no Windows ROCm build to fall back to.
    """

    def setUp(self):
        from pipertrainer import pins as pins_mod

        self.raw = pins_mod.load().raw["torch"]["rocm"]
        self.windows = self.raw.get("windows")
        self.assertIsNotNone(self.windows, "[torch.rocm.windows] is missing")

    def test_the_linux_pin_still_uses_an_index(self):
        self.assertTrue(self.raw["index"].startswith("https://"))
        self.assertTrue(self.raw["spec"].startswith("torch=="))

    def test_windows_is_delivered_as_wheel_urls(self):
        wheels = self.windows["wheels"]
        self.assertTrue(wheels)
        for url in wheels:
            self.assertTrue(url.startswith("https://"), url)
            self.assertTrue(url.endswith(".whl"), url)

    def test_the_runtime_is_installed_before_torch(self):
        # torch links against it; ordering is the whole point of the field.
        self.assertTrue(self.windows["prerequisites"])

    def test_the_wheels_match_the_pinned_interpreter(self):
        wanted = self.windows["requires_python"]
        tag = "cp" + wanted.replace(".", "")
        for url in self.windows["wheels"]:
            self.assertIn(tag, url, f"{url} is not a {wanted} wheel")

    def test_the_wheels_are_for_windows(self):
        for url in self.windows["wheels"]:
            self.assertIn("win_amd64", url, url)

    def test_a_driver_version_and_a_doc_link_are_recorded(self):
        self.assertTrue(self.windows["driver"])
        self.assertTrue(self.windows["docs"].startswith("https://"))


class TorchPinTest(unittest.TestCase):
    """The two delivery mechanisms, as the installer sees them."""

    def pin(self, **kwargs):
        from pipertrainer import pins as pins_mod

        return pins_mod.TorchPin(**kwargs)

    def test_an_index_pin_is_not_url_delivered(self):
        pin = self.pin(index="https://example/whl/cu128", spec="torch==2.9.1")
        self.assertFalse(pin.from_urls)
        self.assertEqual(pin.version, "2.9.1")

    def test_a_url_pin_reports_the_version_from_the_filename(self):
        pin = self.pin(
            wheels=(
                "https://repo.radeon.com/x/torch-2.9.1%2Brocm7.2.1"
                "-cp312-cp312-win_amd64.whl",
            )
        )
        self.assertTrue(pin.from_urls)
        # %2B is an encoded '+': the local version has to survive, because the
        # constraint file records exactly what got installed.
        self.assertEqual(pin.version, "2.9.1+rocm7.2.1")

    def test_the_installed_pin_matches_the_platform(self):
        from pipertrainer import pins as pins_mod

        pin = pins_mod.load().torch("rocm")
        if sys.platform == "win32":
            self.assertTrue(pin.from_urls)
            self.assertEqual(pin.requires_python, "3.12")
            # One published build, so there is no menu of alternatives.
            self.assertEqual(pins_mod.load().torch_alternatives("rocm"), [])
        else:
            self.assertFalse(pin.from_urls)
            self.assertTrue(pins_mod.load().torch_alternatives("rocm"))

    def test_cuda_and_cpu_are_index_delivered_on_both_platforms(self):
        from pipertrainer import pins as pins_mod

        for vendor in ("cuda", "cpu"):
            self.assertFalse(pins_mod.load().torch(vendor).from_urls, vendor)


class AmdOnWindowsTest(unittest.TestCase):
    """An AMD GPU on Windows selects ROCm, not a CPU fallback."""

    def test_an_amd_card_on_windows_is_a_rocm_target(self):
        with mock.patch.object(env_mod, "WINDOWS", True), \
             mock.patch.object(env_mod, "has_nvidia", return_value=False), \
             mock.patch.object(env_mod, "has_amd", return_value=True):
            vendor, notes = env_mod.detect_vendor()
        self.assertEqual(vendor, "rocm")
        # The extra requirements are surfaced, not silently assumed.
        self.assertTrue(any("3.12" in note for note in notes))

    def test_the_note_is_built_from_the_pins(self):
        note = env_mod.rocm_windows_note()
        self.assertIn("3.12", note)
        self.assertIn("26.2.2", note)

    def test_no_gpu_on_windows_still_falls_back_to_cpu(self):
        with mock.patch.object(env_mod, "WINDOWS", True), \
             mock.patch.object(env_mod, "has_nvidia", return_value=False), \
             mock.patch.object(env_mod, "has_amd", return_value=False):
            vendor, _ = env_mod.detect_vendor()
        self.assertEqual(vendor, "cpu")


class HardwareProfileRemovalTest(unittest.TestCase):
    """runtime.hardware is gone; profiles that still carry it must still load."""

    def test_the_schema_has_no_hardware_setting(self):
        paths_seen = {path for path, _, _ in profile_mod.iter_specs(profile_mod.Profile())}
        self.assertNotIn("runtime.hardware", paths_seen)

    def test_an_old_profile_loads_without_complaining_about_hardware(self):
        prof, warnings = profile_mod.from_dict(
            {"schema": 1, "runtime": {"hardware": "bc250", "offline": True}}
        )
        # Written by this tool, so warning about it would blame the user for
        # our own schema change.
        self.assertEqual([w for w in warnings if "hardware" in w], [])
        self.assertTrue(prof.runtime.offline)

    def test_migration_bumps_the_schema_version(self):
        migrated = profile_mod.migrate({"schema": 1, "runtime": {"hardware": "bc250"}})
        self.assertEqual(migrated["schema"], profile_mod.SCHEMA_VERSION)
        self.assertNotIn("hardware", migrated["runtime"])

    def test_a_genuinely_unknown_setting_still_warns(self):
        _, warnings = profile_mod.from_dict({"runtime": {"nonsense": 1}})
        self.assertTrue(any("nonsense" in w for w in warnings))

    def test_runtime_env_survives_as_the_escape_hatch(self):
        prof, _ = profile_mod.from_dict(
            {"runtime": {"env": {"HSA_ENABLE_SDMA": "0"}}}
        )
        self.assertEqual(prof.runtime.env["HSA_ENABLE_SDMA"], "0")

    def test_the_hardware_module_is_gone(self):
        with self.assertRaises(ImportError):
            __import__("pipertrainer.hardware")


class InterpreterPinTest(unittest.TestCase):
    """Windows has no versioned interpreter on PATH; `py -3.13` is two words."""

    def test_candidates_are_argv_lists(self):
        from pipertrainer import pins as pins_mod

        for candidate in pins_mod.load().python_prefer:
            self.assertIsInstance(candidate, list)
            self.assertTrue(candidate, "an empty argv cannot be executed")
            self.assertIsInstance(candidate[0], str)

    def test_windows_selects_versions_through_the_py_launcher(self):
        from pipertrainer import pins as pins_mod

        raw = pins_mod.load().raw["python"]
        self.assertIn("prefer_windows", raw)
        launcher = [c for c in raw["prefer_windows"] if len(c) > 1]
        self.assertTrue(launcher, "no versioned Windows candidate")
        for candidate in launcher:
            self.assertEqual(candidate[0], "py")
            self.assertTrue(candidate[1].startswith("-3."), candidate)


if __name__ == "__main__":
    unittest.main()
