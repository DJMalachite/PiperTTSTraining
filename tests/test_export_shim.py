"""The ONNX export has to run on the exporter upstream was written for.

piper1-gpl calls `torch.onnx.export` with no `dynamo` argument and with
`dynamic_axes`, which is legacy-exporter API. torch 2.9 flipped the default to
`dynamo=True`, and VITS does not survive `torch.export`: an assert over a
boolean-mask-indexed tensor in `vits/transforms.py` becomes a guard on an
unbacked symint. Forcing the TorchScript path is our workaround, and these
tests exist so nobody quietly routes the export back at upstream's module.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from . import _support  # noqa: F401

from pipertrainer.train import export as export_mod
from pipertrainer.train import export_shim


class EntrypointTest(unittest.TestCase):
    def test_export_goes_through_our_shim(self):
        argv = export_mod.export_command(
            "python", Path("/runs/last.ckpt"), Path("/voices/v.onnx")
        )
        self.assertIn("pipertrainer.train.export_shim", argv)

    def test_it_does_not_call_upstream_directly(self):
        argv = export_mod.export_command(
            "python", Path("/runs/last.ckpt"), Path("/voices/v.onnx")
        )
        self.assertNotIn("piper.train.export_onnx", argv)

    def test_upstream_flags_are_unchanged(self):
        # The shim delegates to upstream's main(), so argv must stay its argv.
        ckpt, onnx = Path("/runs/last.ckpt"), Path("/voices/v.onnx")
        argv = export_mod.export_command("python", ckpt, onnx)
        self.assertEqual(argv[argv.index("--checkpoint") + 1], str(ckpt))
        self.assertEqual(argv[argv.index("--output-file") + 1], str(onnx))

    def test_it_runs_as_a_module(self):
        argv = export_mod.export_command("py", Path("a"), Path("b"))
        self.assertEqual(argv[:2], ["py", "-m"])


class WrapExportTest(unittest.TestCase):
    """The pure half of the shim, exercised without torch installed."""

    def test_dynamo_is_forced_off(self):
        seen = {}

        def fake_export(model, *, dynamo=True, **kwargs):
            seen["dynamo"] = dynamo
            return "exported"

        wrapped = export_shim.wrap_export(fake_export)
        self.assertIsNotNone(wrapped)
        self.assertEqual(wrapped("model"), "exported")
        self.assertIs(seen["dynamo"], False)

    def test_it_overrides_an_explicit_true(self):
        # Belt and braces: upstream does not pass dynamo today, but if it
        # started passing True we would still need the legacy path.
        seen = {}

        def fake_export(model, *, dynamo=True):
            seen["dynamo"] = dynamo

        export_shim.wrap_export(fake_export)("model", dynamo=True)
        self.assertIs(seen["dynamo"], False)

    def test_other_arguments_pass_through(self):
        seen = {}

        def fake_export(model, *, dynamo=True, opset_version=None, dynamic_axes=None):
            seen.update(opset=opset_version, axes=dynamic_axes)

        export_shim.wrap_export(fake_export)(
            "model", opset_version=15, dynamic_axes={"input": {0: "batch"}}
        )
        self.assertEqual(seen["opset"], 15)
        self.assertEqual(seen["axes"], {"input": {0: "batch"}})

    def test_a_torch_without_dynamo_needs_no_wrapper(self):
        # torch < 2.5 has only the TorchScript exporter; passing the argument
        # would be a TypeError.
        def old_export(model, *, opset_version=None):
            return None

        self.assertIsNone(export_shim.wrap_export(old_export))

    def test_an_uninspectable_callable_is_left_alone(self):
        self.assertIsNone(export_shim.wrap_export(print))

    def test_the_module_imports_without_torch(self):
        # It is imported by our CLI's test suite on machines with no torch;
        # every torch import inside it must stay function-local.
        self.assertTrue(callable(export_shim.main))


if __name__ == "__main__":
    unittest.main()
