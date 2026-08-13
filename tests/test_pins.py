"""pins.toml is the single source of truth, so its shape is worth asserting.

The `[build]` section is the interesting one. It looks redundant with pip's
build isolation and is not: `build_ext` runs piper1-gpl's setup.py directly,
outside any isolated environment, so `from skbuild import setup` fails with a
ModuleNotFoundError whose obvious fix (a distro package named "skbuild") does
not exist. Deleting the section would put that back.
"""

from __future__ import annotations

import unittest

from . import _support  # noqa: F401

from pipertrainer import pins as pins_mod


class BuildRequiresTest(unittest.TestCase):
    def setUp(self):
        self.pins = pins_mod.load()

    def test_build_requires_is_present_and_non_empty(self):
        self.assertTrue(self.pins.build_requires)

    def test_scikit_build_is_required(self):
        # Upstream's setup.py imports skbuild; nothing else supplies it once
        # pip's build environment for the editable install is discarded.
        names = [req.split(">")[0].split("=")[0].split("<")[0] for req in self.pins.build_requires]
        self.assertIn("scikit-build", names)

    def test_cmake_and_ninja_are_required(self):
        # skbuild shells out to both, and the espeak-ng ExternalProject needs
        # a cmake newer than several distros ship.
        joined = " ".join(self.pins.build_requires)
        self.assertIn("cmake", joined)
        self.assertIn("ninja", joined)

    def test_scikit_build_core_is_not_what_upstream_imports(self):
        # `from skbuild import setup` is classic scikit-build. scikit-build-core
        # provides no `skbuild` module, so swapping them would not fix anything.
        self.assertNotIn("scikit-build-core", self.pins.build_requires)


class ExportRequiresTest(unittest.TestCase):
    """torch.onnx.export needs onnxscript, and nothing else declares it.

    piper1-gpl does not list it in [train], and torch 2.9 imports it
    unconditionally from torch.onnx. Without this section, export fails with
    ModuleNotFoundError after training has already completed.
    """

    def setUp(self):
        self.pins = pins_mod.load()

    def test_export_requires_is_present_and_non_empty(self):
        self.assertTrue(self.pins.export_requires)

    def test_onnxscript_is_required(self):
        joined = " ".join(self.pins.export_requires)
        self.assertIn("onnxscript", joined)

    def test_it_is_pinned_rather_than_floating(self):
        # Same reasoning as the whisper pin: an unpinned resolve is how a
        # dependency change arrives without anyone deciding on it.
        for req in self.pins.export_requires:
            with self.subTest(req=req):
                self.assertIn("==", req)

    def test_torch_is_not_pulled_in_here(self):
        # onnxscript has no torch dependency; naming one here would be a way
        # to replace a vendor wheel by accident.
        joined = " ".join(self.pins.export_requires)
        self.assertNotIn("torch", joined)


class InstallStepTest(unittest.TestCase):
    def test_export_deps_runs_after_the_torch_constraint(self):
        # Installing before the constraint file exists is what lets pip
        # resolve a different torch.
        from pipertrainer import install

        names = [step.name for step in install.STEPS]
        self.assertIn("export_deps", names)
        self.assertGreater(names.index("export_deps"), names.index("constraint"))

    def test_export_deps_runs_before_verify(self):
        from pipertrainer import install

        names = [step.name for step in install.STEPS]
        self.assertLess(names.index("export_deps"), names.index("verify"))


class TorchPinTest(unittest.TestCase):
    def setUp(self):
        self.pins = pins_mod.load()

    def test_every_vendor_is_installable_one_way_or_the_other(self):
        # Two delivery mechanisms, and a pin must be exactly one of them: an
        # index that replaces PyPI, or explicit wheel URLs. AMD's native
        # Windows ROCm build is the second kind — there is no index for it.
        for vendor in self.pins.vendors:
            with self.subTest(vendor=vendor):
                pin = self.pins.torch(vendor)
                if pin.from_urls:
                    for url in pin.wheels:
                        self.assertTrue(url.startswith("https://"), url)
                    self.assertFalse(pin.index, "a URL pin has no index")
                else:
                    self.assertTrue(pin.index.startswith("https://"))
                    self.assertTrue(pin.spec.startswith("torch=="))

    def test_the_linux_pins_are_all_index_delivered(self):
        # Read from the raw table so this holds when the suite runs on Windows.
        for vendor in self.pins.vendors:
            entry = self.pins.raw["torch"][vendor]
            with self.subTest(vendor=vendor):
                self.assertTrue(entry["index"].startswith("https://"))
                self.assertTrue(entry["spec"].startswith("torch=="))

    def test_unknown_vendor_is_an_error(self):
        with self.assertRaises(pins_mod.PinsError):
            self.pins.torch("mali")


if __name__ == "__main__":
    unittest.main()
