#!/bin/bash
#SBATCH --job-name=tn_featstore
#SBATCH --account=gogpt
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/kfs2/projects/gogpt/Ebadmus/logs/featstore-%j.out
#
# Rebuild a feature-space scenario corpus from the pu scenario stores.
#
# This exists because the v4..v8 feature stores were built by ad-hoc commands that
# nothing recorded, so deleting them destroyed the only description of how to make
# them again. Every future store comes from this file.
#
# Three stages, fail-closed:
#   1. fit    -- one leakage-safe global scaler from TRAIN-split feeders only
#                (train membership comes from gridfm/topology_fingerprints.json,
#                 so structurally identical feeders can never straddle the split)
#   2. featurize -- rewrite pu stores into feature-space stores under OUT
#   3. gate   -- re-derive the conversion from the pu source and require an exact match
#
# The scaler is written into OUT, never into the source corpus: training_data/ is
# read-only input and must stay that way.
#
# Usage:
#   sbatch scripts/build_feature_store.sbatch                    # full v8 store
#   SMOKE=1 sbatch scripts/build_feature_store.sbatch            # 8 feeders/corpus
#   OUT=/path/to/store sbatch scripts/build_feature_store.sbatch
set -euo pipefail

PROJECT=/kfs2/projects/gogpt/Ebadmus
cd "${PROJECT}/Training_new"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# datakit is a sibling repo on PYTHONPATH, not pip-installed. Stage 1 patches sys.path
# itself, but `-m gridfm.featurizing` imports gridfm/__init__.py -> datakit and needs
# this set for real.
export PYTHONPATH="${PROJECT}:${PROJECT}/Training_new${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT}/.venv-train/bin/python}"
SRC="${SRC:-${PROJECT}/training_data}"
SMOKE="${SMOKE:-0}"
if [[ "${SMOKE}" == "1" ]]; then
    # Smoke output goes under scratch/ and deletes itself on success, so a throwaway
    # store never survives next to the real corpora. On failure it is kept and its path
    # printed, because that is when you need to look inside it.
    OUT="${OUT:-${PROJECT}/Training_new/scratch/featstore_smoke_${SLURM_JOB_ID:-$$}}"
    PER_CORPUS="${PER_CORPUS:-8}"
    trap 'rc=$?; if [[ $rc -eq 0 ]]; then rm -rf "${OUT}" "${WORKDIR}"; echo "[smoke] cleaned up ${OUT}"; else echo "[smoke] FAILED (rc=$rc); kept for inspection: ${OUT}" >&2; fi' EXIT
else
    OUT="${OUT:-${PROJECT}/training_data_foundation_v8_fractional}"
    PER_CORPUS="${PER_CORPUS:-0}"
fi
WORKERS="${WORKERS:-${SLURM_CPUS_ON_NODE:-$(nproc)}}"
WORKERS="${WORKERS%%(*}"

[[ -d "${SRC}" ]] || { echo "no source corpus: ${SRC}" >&2; exit 2; }
mkdir -p "${OUT}"
# The scaler is fitted into a staging dir, not into OUT: stage 2 installs it into the
# store itself, and fitting straight into OUT makes that copy a self-copy (SameFileError).
# The staging dir doubles as the run's cache_dir.
WORKDIR="${WORKDIR:-${PROJECT}/Training_new/cache/$(basename "${OUT}")}"
mkdir -p "${WORKDIR}"
SCALER="${WORKDIR}/feature_scaler.json"

# A bare feeder count is the cheapest proof the source is really there before we
# spend a node on it.
echo "[$(date '+%F %T')] src=${SRC} out=${OUT} workers=${WORKERS} smoke=${SMOKE}"

LIMIT_ARGS=()
if [[ "${PER_CORPUS}" != "0" ]]; then
    LIMIT_ARGS=(--limit-per-corpus "${PER_CORPUS}")
fi

echo "[$(date '+%F %T')] stage 1/3: fit leakage-safe scaler"
"${PYTHON_BIN}" -u scripts/fit_scenario_scaler.py \
    --root "${SRC}" \
    --out "${SCALER}" \
    --workers "${WORKERS}" \
    "${LIMIT_ARGS[@]}"

echo "[$(date '+%F %T')] stage 2/3: rewrite pu stores into feature space"
"${PYTHON_BIN}" -u -m gridfm.featurizing scenario-stores \
    --root "${SRC}" \
    --scaler "${SCALER}" \
    --out-root "${OUT}" \
    --unified-line-scale \
    --workers "${WORKERS}" \
    "${LIMIT_ARGS[@]}"

# The gate checks EVERY feeder by default. A sampled gate is worse than useless here:
# it demands all nine exact families decode a row, but rare families are concentrated in
# a handful of feeders (in dss_data, generator lives in 3 of 77, storage in 4), so a small
# sample reports "no rows were decoded" for data that is entirely correct. Per-feeder
# exact decode is cached on disk, so gating the whole corpus is affordable.
echo "[$(date '+%F %T')] stage 3/3: exact-conversion + exact-Y contract gate"
"${PYTHON_BIN}" -u scripts/check_feature_store_contract.py \
    --root "${OUT}" \
    --source-root "${SRC}" \
    --limit "${GATE_LIMIT:-1000000}" \
    --workers "${WORKERS}"

echo "[$(date '+%F %T')] feature store COMPLETE: ${OUT}"
