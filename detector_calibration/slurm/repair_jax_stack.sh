#!/bin/bash
# One-shot repair for the user-site jax/jaxlib/cudnn ABI mismatch that broke
# logs/bo-29719387.out. Runs interactively (no sbatch needed). Idempotent.
set -euo pipefail

SIF_FILE=${SIF_FILE:-/sdf/group/neutrino/pgranger/larnd-sim-jax.sif}

if [[ ! -f "$SIF_FILE" ]]; then
  echo "FATAL: SIF not found at $SIF_FILE" >&2
  exit 1
fi
echo "Using SIF: $SIF_FILE"

apptainer exec --nv \
  -B /sdf,/fs,/sdf/scratch,/lscratch \
  "$SIF_FILE" /bin/bash <<'INSIDE'
set -euo pipefail

SP=/sdf/home/c/cyifan/.local/lib/python3.10/site-packages
echo "==> sweeping orphan partial-uninstalls in $SP"
cd "$SP"
shopt -s nullglob
for d in '~larndsim'* '~-rndsim'* '~~rndsim'*; do
  echo "  rm -rf $d"
  rm -rf -- "$d"
done
# Also drop the accidentally-installed larndsim copy (we sys.path-inject the repo instead).
for d in larndsim larndsim-*.dist-info; do
  if [[ -e "$d" ]]; then
    echo "  rm -rf $d"
    rm -rf -- "$d"
  fi
done
shopt -u nullglob

echo "==> reinstalling jax[cuda12]==0.6.2 (pulls matching jaxlib + plugin + cudnn)"
pip3 install --user --upgrade --force-reinstall "jax[cuda12]==0.6.2"

echo "==> smoke test"
python3 - <<'PY'
import jax, jaxlib, jax.numpy as jnp
print("jax    :", jax.__version__)
print("jaxlib :", jaxlib.__version__)
print("devices:", jax.devices())
print("op test:", float(jnp.sum(jnp.arange(10))))
PY
INSIDE
