#!/bin/bash
#SBATCH -J t22_fixed_y
#SBATCH -A view
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 1:00:00
#SBATCH -o /kfs2/projects/gogpt/Ebadmus/logs/t22-fixed-y-%j.out
#SBATCH -e /kfs2/projects/gogpt/Ebadmus/logs/t22-fixed-y-%j.err

set -euo pipefail
# datakit is a sibling repo on PYTHONPATH, not pip-installed.
export PYTHONPATH=/kfs2/projects/gogpt/Ebadmus:/kfs2/projects/gogpt/Ebadmus/Training_new${PYTHONPATH:+:${PYTHONPATH}}

ROOT="/kfs2/projects/gogpt/Ebadmus"
TD="${ROOT}/training_data"
DATA="${ROOT}/data"
PY="${ROOT}/.venv/bin/python"
AUDIT_ROOT="${AUDIT_ROOT:-${ROOT}/Training_new}"
RUN_ROOT="${AUDIT_ROOT}/runs/t22_fixed_y_seed260719_v2"
SOURCE_ROOT="${RUN_ROOT}/source"
STORE_ROOT="${RUN_ROOT}/stores"
CORPORA=(SMART-DS_1000 new_dss_data dss_data minimal_component)
PER_CORPUS=3
COUNT=40

if [[ -e "${RUN_ROOT}" ]]; then
  echo "refusing to overwrite existing control: ${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${SOURCE_ROOT}" "${STORE_ROOT}"

# Reproduce T22's spread selection exactly, then copy only those feeder source
# trees into an isolated control root.  The production corpus and raw data are
# read-only inputs to this experiment.
for corpus in "${CORPORA[@]}"; do
  stores=("${TD}/${corpus}"/*/static.pt)
  if [[ "${#stores[@]}" -eq 0 ]]; then
    echo "no scenario stores for ${corpus}" >&2
    exit 2
  fi
  step=$(( ${#stores[@]} / PER_CORPUS ))
  if [[ "${step}" -lt 1 ]]; then step=1; fi
  picked=0
  for ((index=0; index<${#stores[@]} && picked<PER_CORPUS; index+=step)); do
    store_name="$(basename "$(dirname "${stores[$index]}")")"
    # Some source corpora were already arranged into flat, collision-free
    # feeder directories; SMART-DS retains its hierarchy. Prefer the exact
    # flat name, then invert the hierarchy separator used by scenario stores.
    relative="${store_name}"
    source_feeder="${DATA}/${corpus}/${relative}"
    if [[ ! -f "${source_feeder}/Master.dss" && ! -f "${source_feeder}/network/Master.dss" ]]; then
      relative="${store_name//__//}"
      source_feeder="${DATA}/${corpus}/${relative}"
    fi
    staged_feeder="${SOURCE_ROOT}/${corpus}/${relative}"
    if [[ ! -f "${source_feeder}/Master.dss" && ! -f "${source_feeder}/network/Master.dss" ]]; then
      echo "cannot map store ${store_name} to source feeder ${source_feeder}" >&2
      exit 2
    fi
    mkdir -p "$(dirname "${staged_feeder}")"
    cp -a "${source_feeder}" "${staged_feeder}"
    printf '%s\t%s\t%s\n' "${corpus}" "${store_name}" "${source_feeder}" | tee -a "${RUN_ROOT}/selection.tsv"
    picked=$((picked + 1))
  done
done

AVAILABLE_CORES="${SLURM_CPUS_ON_NODE:-$(nproc)}"
AVAILABLE_CORES="${AVAILABLE_CORES%%(*}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

"${PY}" -u "${ROOT}/datakit/pipelines/make_training_corpora_pt.py" \
  --source-root "${SOURCE_ROOT}" \
  --out-root "${STORE_ROOT}" \
  --corpora "${CORPORA[@]}" \
  --count "${COUNT}" \
  --mode loading \
  --seed 260719 \
  --workers "${AVAILABLE_CORES}" \
  --attempts 256 \
  --first-variant-attempts 3 \
  --float-dtype float64 \
  --validate-stores \
  --fail-on-skipped-feeders \
  --strict-exit

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
"${PY}" -u "${AUDIT_ROOT}/gridfm/probes/y_recovery_audit.py" \
  --root "${STORE_ROOT}" \
  --corpora "${CORPORA[@]}" \
  --per-corpus "${PER_CORPUS}" \
  --k 1 2 4 8 16 32 \
  --holdout 39 \
  --output "${RUN_ROOT}/fixed_y_audit.json"
