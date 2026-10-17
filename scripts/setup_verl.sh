#!/usr/bin/env bash
# Clone verl v0.5.0, apply the Alpha-RTL patch, and link the recipes into it.
# Usage: bash scripts/setup_verl.sh [VERL_DIR]   (default: ./verl)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
VERL_DIR=${1:-$ROOT/verl}

if [ ! -d "$VERL_DIR" ]; then
  git clone --branch v0.5.0 --depth 1 https://github.com/volcengine/verl.git "$VERL_DIR"
fi
git -C "$VERL_DIR" apply --check "$ROOT/patches/verl-v0.5.0.patch" 2>/dev/null \
  && git -C "$VERL_DIR" apply "$ROOT/patches/verl-v0.5.0.patch" \
  || echo "[setup] patch already applied or does not apply cleanly; check $VERL_DIR"

for r in ttt_math_construct ttt_rtl ttt_c910_fpu; do
  ln -sfn "$ROOT/recipe/$r" "$VERL_DIR/recipe/$r"
done
echo "[setup] verl ready at $VERL_DIR"
