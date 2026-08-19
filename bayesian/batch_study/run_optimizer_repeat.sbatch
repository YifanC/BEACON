#!/bin/bash
#SBATCH --job-name=bo-repeat
#SBATCH --partition=ampere
#SBATCH --account=neutrino:default@ampere
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00

set -euo pipefail

: "${BAYESIAN_DATA_SIZE:?set BAYESIAN_DATA_SIZE to 1000cm or 2000cm}"
: "${BAYESIAN_REPEAT:?set BAYESIAN_REPEAT to 1 or 2}"
: "${BAYESIAN_OPTIMIZER_SEED:?set BAYESIAN_OPTIMIZER_SEED}"

BAYESIAN=/sdf/home/i/iatif/REAL_BEACON/bayesian
UPSTREAM=/sdf/home/i/iatif/larnd-sim-jax
CONTAINER=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif

case "$BAYESIAN_DATA_SIZE" in
  1000cm)
    TARGET="$BAYESIAN/.local/two_d/target.npz"
    EXPECTED_TARGET_HASH=ed72e051a6111cfee218e0e009a9ebd78ae1167ea7225a58f2ac4b32f6edd34f
    PHYSICAL_LENGTH_CM=1000
    N_EVENTS=176
    ;;
  2000cm)
    TARGET="$BAYESIAN/batch_study/2000cm/target_2000cm.npz"
    EXPECTED_TARGET_HASH=c2b7c0a7c2c370efe6af34c02acc248a9ce04bb44d4176f09b1752ecb4d8503d
    PHYSICAL_LENGTH_CM=2000
    N_EVENTS=-1
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export LARNDSIM_MEMORY_EFFICIENT_SIGNALS=1
    export LARNDSIM_SIGNAL_CONTRIB_BLOCK_SIZE=500000
    export LARNDSIM_PROBABILISTIC_BLOCK_SIZE=128
    ;;
  *) echo "unsupported data size: $BAYESIAN_DATA_SIZE" >&2; exit 2 ;;
esac

case "$BAYESIAN_REPEAT" in 1|2) ;; *) echo "repeat must be 1 or 2" >&2; exit 2 ;; esac
ACTUAL_TARGET_HASH=$(sha256sum "$TARGET" | awk '{print $1}')
test "$ACTUAL_TARGET_HASH" = "$EXPECTED_TARGET_HASH"

EXPERIMENT_DIR="$BAYESIAN/batch_study/$BAYESIAN_DATA_SIZE/repeat_$BAYESIAN_REPEAT"
RUN_ROOT="$EXPERIMENT_DIR/history"
test ! -e "$RUN_ROOT"
mkdir -p "$EXPERIMENT_DIR/config" "$EXPERIMENT_DIR/logs" \
         "$EXPERIMENT_DIR/results" "$EXPERIMENT_DIR/plots" \
         "$RUN_ROOT/raw/traces" "$RUN_ROOT/raw/initial_design" \
         "$RUN_ROOT/raw/config" "$RUN_ROOT/raw/checkpoints" \
         "$RUN_ROOT/raw/simulator" "$RUN_ROOT/raw/logs"

export BAYESIAN_EXPERIMENT="${BAYESIAN_DATA_SIZE}-repeat-${BAYESIAN_REPEAT}"
export BAYESIAN_RUN_ROOT="$RUN_ROOT"
export BAYESIAN_TARGET_NPZ="$TARGET"
export BAYESIAN_PHYSICAL_LENGTH_CM="$PHYSICAL_LENGTH_CM"
export BAYESIAN_N_EVENTS="$N_EVENTS"
export BAYESIAN_MAX_NBATCH=1
export BAYESIAN_SIMULATOR_SEED=0
export BAYESIAN_DATA_SEED=0
export PYTHONPATH="$BAYESIAN/workflows:$BAYESIAN/workflows/six_d:$UPSTREAM:$UPSTREAM/src"

printf 'data_size=%s repeat=%s optimizer_seed=%s target=%s target_hash=%s output=%s\n' \
  "$BAYESIAN_DATA_SIZE" "$BAYESIAN_REPEAT" "$BAYESIAN_OPTIMIZER_SEED" \
  "$TARGET" "$ACTUAL_TARGET_HASH" "$EXPERIMENT_DIR"

cd "$UPSTREAM"
run_python() {
  singularity exec --nv --bind /sdf:/sdf "$CONTAINER" python3 "$@"
}

run_python "$BAYESIAN/workflows/six_d/build_initial_design.py"
run_python "$BAYESIAN/workflows/six_d/run_bo.py"
run_python "$BAYESIAN/batch_study/prepare_snapshot.py" --run-root "$RUN_ROOT" --stage bo1
run_python "$BAYESIAN/workflows/six_d/continue_bo.py" --mode trust_region
run_python "$BAYESIAN/batch_study/prepare_snapshot.py" --run-root "$RUN_ROOT" --stage botr
run_python "$BAYESIAN/workflows/six_d/continue_bo.py" --mode trust_region_2
run_python "$BAYESIAN/batch_study/analyze_experiment.py" --experiment-dir "$EXPERIMENT_DIR"
