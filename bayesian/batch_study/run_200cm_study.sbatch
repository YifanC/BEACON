#!/bin/bash
#SBATCH --job-name=bo-200cm-study
#SBATCH --partition=ampere
#SBATCH --account=neutrino:default@ampere
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1%1
#SBATCH --output=/sdf/home/i/iatif/REAL_BEACON/bayesian/batch_study/shared/logs/run-%A_%a.out

set -euo pipefail

BAYESIAN=/sdf/home/i/iatif/REAL_BEACON/bayesian
UPSTREAM=/sdf/home/i/iatif/larnd-sim-jax
CONTAINER=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif

if [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]; then
  EXPERIMENT=200cm_seed0
  SIMULATOR_SEED=0
else
  EXPERIMENT=200cm_seed1
  SIMULATOR_SEED=1
fi

EXPERIMENT_DIR="$BAYESIAN/batch_study/$EXPERIMENT"
RUN_ROOT="$EXPERIMENT_DIR/history"
mkdir -p "$EXPERIMENT_DIR/config" "$EXPERIMENT_DIR/logs" \
         "$EXPERIMENT_DIR/history" "$EXPERIMENT_DIR/results" \
         "$EXPERIMENT_DIR/plots" \
         "$RUN_ROOT/raw/traces" "$RUN_ROOT/raw/initial_design" \
         "$RUN_ROOT/raw/config" "$RUN_ROOT/raw/checkpoints" \
         "$RUN_ROOT/raw/simulator" "$RUN_ROOT/raw/logs"
exec > >(tee "$EXPERIMENT_DIR/logs/workflow-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log") 2>&1

export BAYESIAN_EXPERIMENT="$EXPERIMENT"
export BAYESIAN_RUN_ROOT="$RUN_ROOT"
export BAYESIAN_TARGET_NPZ="$BAYESIAN/batch_study/shared/target_200cm.npz"
export BAYESIAN_PHYSICAL_LENGTH_CM=200
export BAYESIAN_N_EVENTS=-1
export BAYESIAN_MAX_NBATCH=1
export BAYESIAN_SIMULATOR_SEED="$SIMULATOR_SEED"
export BAYESIAN_DATA_SEED=0
export PYTHONPATH="$BAYESIAN/workflows:$BAYESIAN/workflows/six_d:$UPSTREAM:$UPSTREAM/src"

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
