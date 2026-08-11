"""Run ``piper.train.export_onnx`` against the exporter it was written for.

Upstream calls ``torch.onnx.export`` without a ``dynamo`` argument, on
``v1.6.0`` and on ``main`` alike, and passes ``dynamic_axes`` — which is
legacy-exporter API. torch 2.9 flipped the default to ``dynamo=True``, routing
the call through ``torch.export.export``, which fails on VITS::

    File "vits/transforms.py", line 174, in rational_quadratic_spline
        assert (discriminant >= 0).all(), discriminant
    GuardOnDataDependentSymNode: Could not guard on data-dependent expression
    Eq(u2, 1)

The assert is over a tensor produced by boolean-mask indexing
(``inputs[inside_interval_mask]``), so its size is an unbacked symint.
``torch.export`` has to resolve ``.all()`` to a concrete bool at trace time and
cannot. Nothing is wrong with the checkpoint: the same graph traces fine
through the TorchScript exporter, which is what every Piper voice to date was
exported with.

We cannot edit ``piper1-gpl``, so this module wraps ``torch.onnx.export`` to
force ``dynamo=False`` and then hands over to upstream's ``main()`` untouched —
same argv, same everything else. Delete it when upstream passes ``dynamo``
itself, or when the model stops tripping ``torch.export``.
"""

from __future__ import annotations

import inspect
import sys


def wrap_export(original):
    """``original`` with ``dynamo=False`` forced.

    Returns ``None`` when no wrapper is needed or possible: a torch too old to
    know the argument (< 2.5) has only the TorchScript exporter anyway, and a
    callable we cannot introspect is left alone rather than guessed at.
    """
    try:
        accepted = inspect.signature(original).parameters
    except (TypeError, ValueError):  # pragma: no cover - C-implemented callable
        return None
    if "dynamo" not in accepted:
        return None

    def export(*args, **kwargs):
        kwargs["dynamo"] = False
        return original(*args, **kwargs)

    return export


def _force_legacy_exporter() -> None:
    import torch.onnx

    wrapped = wrap_export(torch.onnx.export)
    if wrapped is not None:
        torch.onnx.export = wrapped  # type: ignore[assignment]


def main() -> None:
    _force_legacy_exporter()
    from piper.train.export_onnx import main as upstream_main

    upstream_main()


if __name__ == "__main__":
    main()
