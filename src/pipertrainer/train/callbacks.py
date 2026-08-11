"""Adjusting upstream's ModelCheckpoints from outside piper1-gpl.

Upstream passes its two ``ModelCheckpoint``s through ``trainer_defaults``
(``train/__main__.py:65``). ``LightningCLI._instantiate_trainer`` *concatenates*
config callbacks onto those rather than replacing them::

    config[key].extend(callbacks)
    if key in self.trainer_defaults:
        config[key] += self.trainer_defaults[key]

So there is no configuration that removes or reconfigures them. Listing a
replica adds a third callback; listing an *identical* replica is a hard error,
because two stateful callbacks of one type may not share a ``state_key``:

    RuntimeError: Found more than one stateful callback of type
    `ModelCheckpoint`

What can be done is reach the instances at runtime. A callback named in the
config is constructed alongside upstream's and can adjust them in ``setup``,
which runs long before the first epoch ends. That is all this module is.

The policy logic is a pure function over anything carrying ``monitor`` and
``save_top_k``, so it is testable without lightning installed; only the thin
``Callback`` subclass needs the real thing.
"""

from __future__ import annotations

from typing import Any, Sequence

try:  # pragma: no cover - exercised only inside the training venv
    from lightning.pytorch import Callback as _Callback
except ImportError:  # pragma: no cover - our CLI never imports lightning
    _Callback = object


def apply_policy(
    checkpoints: Sequence[Any],
    *,
    save_top_k: int | None = None,
    disable_monitors: Sequence[str] = (),
) -> list[str]:
    """Retune upstream's checkpoint callbacks in place.

    ``disable_monitors`` sets ``save_top_k = 0`` on any checkpoint watching one
    of those metrics. Zero is the specific value that matters:
    ``_save_topk_checkpoint`` returns on it *before* testing whether the
    monitored key is present, which is the check that otherwise raises
    ``MisconfigurationException`` at the end of the first epoch.

    Returns a description of every change, because nothing is dropped silently.
    """
    disabled = set(disable_monitors)
    changes: list[str] = []
    for callback in checkpoints:
        monitor = getattr(callback, "monitor", None)
        if monitor in disabled:
            if getattr(callback, "save_top_k", 0) != 0:
                callback.save_top_k = 0
                changes.append(
                    f"checkpoint monitoring {monitor!r} disabled: that metric "
                    f"is never logged in this configuration"
                )
            continue
        if save_top_k is not None and getattr(callback, "save_top_k", None) != save_top_k:
            previous = getattr(callback, "save_top_k", None)
            callback.save_top_k = save_top_k
            changes.append(
                f"checkpoint monitoring {monitor!r}: save_top_k "
                f"{previous} -> {save_top_k}"
            )
    return changes


class CheckpointPolicy(_Callback):  # type: ignore[misc,valid-type]
    """Applies :func:`apply_policy` to whatever checkpoints the trainer has."""

    def __init__(
        self,
        save_top_k: int | None = None,
        disable_monitors: Sequence[str] = (),
    ) -> None:
        super().__init__()
        self.save_top_k = save_top_k
        self.disable_monitors = list(disable_monitors)

    def setup(self, trainer: Any, pl_module: Any, stage: str) -> None:
        if stage != "fit":
            return
        changes = apply_policy(
            trainer.checkpoint_callbacks,
            save_top_k=self.save_top_k,
            disable_monitors=self.disable_monitors,
        )
        for change in changes:
            print(f"[pipertrainer] {change}", flush=True)
