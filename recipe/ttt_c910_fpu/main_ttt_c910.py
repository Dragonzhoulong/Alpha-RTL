"""Hydra entry point for TTT-Discover C910 FPU optimization."""

import os

import hydra

from recipe.ttt_math_construct.algorithms import ensure_ttt_discover_algorithms_registered
from verl.trainer.main_ppo import run_ppo


def _patch_swanlab_local_transaction_bug() -> None:
    """Avoid a swanboard local-mode crash when creating multi-run charts."""
    if os.environ.get("SWANLAB_MODE", "").strip().lower() != "local":
        return

    try:
        import peewee
        import swanboard.callback as swanboard_callback
    except Exception:
        return

    if getattr(swanboard_callback, "_ttt_local_patch_installed", False):
        return

    original_add_multi_chart = swanboard_callback.add_multi_chart

    def safe_add_multi_chart(*args, **kwargs):
        try:
            return original_add_multi_chart(*args, **kwargs)
        except peewee.OperationalError as exc:
            message = str(exc)
            if "cannot commit - no transaction is active" not in message:
                raise
            if not getattr(safe_add_multi_chart, "_warned", False):
                print(
                    "[ttt_c910] Ignoring swanboard local-mode transaction bug in "
                    "add_multi_chart; metric logging will continue."
                )
                safe_add_multi_chart._warned = True
            return None

    swanboard_callback.add_multi_chart = safe_add_multi_chart
    swanboard_callback._ttt_local_patch_installed = True


@hydra.main(config_path="config", config_name="ttt_c910_trainer", version_base=None)
def main(config):
    ensure_ttt_discover_algorithms_registered()
    _patch_swanlab_local_transaction_bug()
    run_ppo(config)


if __name__ == "__main__":
    main()
