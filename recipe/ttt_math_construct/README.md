# TTT Math Construction (Erdos MVP)

This recipe adapts VERL PPO to a TTT-Discover style workflow for the Erdos minimum overlap task.

## What is included

- `config/ttt_erdos_trainer.yaml`: training config with custom dataset, sampler, reward, and advantage estimator.
- `algorithms.py`: recipe-local registration for Discover-style grouped advantage, without editing `verl/trainer/`.
- `dataset.py`: prompt dataset backed by a mutable state pool.
- `sampler.py`: curriculum sampler that performs PUCT state selection and updates from rollout feedback.
- `state_pool.py`: archive of states with lineage-aware blocking and top-k child retention.
- `erdos_reward.py`: code-execution reward function returning continuous rewards and state update metadata.
- `erdos_utils.py`: shared validation and C5 computation utilities.
- `main_ttt_erdos.py`: Hydra entrypoint for this recipe.
- `TTT_DISCOVER_ALGORITHM_FLOW.md`: distilled algorithm flow from `discover/`, mapped to this VERL recipe.

## Quick start

```bash
python3 -m recipe.ttt_math_construct.main_ttt_erdos \
  actor_rollout_ref.model.path=/path/to/model
```

Or use:

```bash
bash recipe/ttt_math_construct/run_ttt_erdos.sh
```

## Notes

- The reward function executes model-generated Python in a subprocess with timeout.
- `algorithm.adv_estimator` is set to `ttt_entropic_adaptive`.
- Constant-reward groups are zeroed at the advantage stage, matching Discover's "remove constant reward groups" effect without changing `verl/trainer/`.
- Actor policy loss uses built-in `gpg`, which is equivalent here to Discover's unclipped `importance_sampling` loss.
- State-pool snapshots are persisted under `${trainer.default_local_dir}/ttt_state_pool`.
- Per-step rollout samples are dumped to `${trainer.default_local_dir}/rollout_samples/<step>.jsonl`, including raw response, parsed code metadata, validity, and improvement flags.
- The curriculum sampler logs recipe-specific `ttt/*` and `puct/*` metrics directly to SwanLab when available.
- When `SWANLAB_MODE=local`, the recipe entrypoint installs a small monkey patch for the current `swanlab/swanboard` local-mode transaction bug so training does not crash while logging.
- Batch defaults are intentionally small for MVP stability and debugging.
