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


class TorchPinTest(unittest.TestCase):
    def setUp(self):
        self.pins = pins_mod.load()

    def test_every_vendor_has_an_index_and_spec(self):
        for vendor in self.pins.vendors:
            with self.subTest(vendor=vendor):
                pin = self.pins.torch(vendor)
                self.assertTrue(pin.index.startswith("https://"))
                self.assertTrue(pin.spec.startswith("torch=="))

    def test_unknown_vendor_is_an_error(self):
        with self.assertRaises(pins_mod.PinsError):
            self.pins.torch("mali")


if __name__ == "__main__":
    unittest.main()
