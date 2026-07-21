#!/bin/bash
#SBATCH -J t22_fixed_pair
#SBATCH -A view
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 1:00:00
#SBATCH -o /kfs2/projects/gogpt/Ebadmus/logs/t22-fixed-pair-%j.out
#SBATCH -e /kfs2/projects/gogpt/Ebadmus/logs/t22-fixed-pair-%j.err

set -euo pipefail

ROOT="/kfs2/projects/gogpt/Ebadmus"
PY="${ROOT}/.venv/bin/python"
DATA="${ROOT}/data"
AUDIT_ROOT="${AUDIT_ROOT:-${ROOT}/Training_new}"
ORIGINAL_JSON="${AUDIT_ROOT}/runs/t22_audit/original.json"
PRIOR_CONTROL="${AUDIT_ROOT}/runs/t22_fixed_y_seed260719_v2/stores"
RUN_ROOT="${AUDIT_ROOT}/runs/t22_fixed_y_paired_seed260719"
SOURCE_ROOT="${RUN_ROOT}/source"
STORE_ROOT="${RUN_ROOT}/stores"
CORPORA=(SMART-DS_1000 new_dss_data dss_data minimal_component)
COUNT=40

if [[ ! -f "${ORIGINAL_JSON}" ]]; then
  echo "missing original T22 audit: ${ORIGINAL_JSON}" >&2
  exit 2
fi
if [[ -e "${RUN_ROOT}" ]]; then
  echo "refusing to overwrite existing paired control: ${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${SOURCE_ROOT}" "${STORE_ROOT}"

# Use exactly the unique feeder stores that survived T22's component-eligibility
# filter. This makes the changing-Y and fixed-Y measurements feeder-paired.
while IFS=$'\t' read -r corpus store_name; do
  relative="${store_name}"
  source_feeder="${DATA}/${corpus}/${relative}"
  if [[ ! -f "${source_feeder}/Master.dss" && ! -f "${source_feeder}/network/Master.dss" ]]; then
    relative="${store_name//__//}"
    source_feeder="${DATA}/${corpus}/${relative}"
  fi
  if [[ ! -f "${source_feeder}/Master.dss" && ! -f "${source_feeder}/network/Master.dss" ]]; then
    echo "cannot map ${corpus}/${store_name} to source data" >&2
    exit 2
  fi
  staged_feeder="${SOURCE_ROOT}/${corpus}/${relative}"
  mkdir -p "$(dirname "${staged_feeder}")"
  cp -a "${source_feeder}" "${staged_feeder}"
  printf '%s\t%s\t%s\n' "${corpus}" "${store_name}" "${source_feeder}" | tee -a "${RUN_ROOT}/selection.tsv"

  # Reuse an identical seed/mode/count control store when the first control
  # happened to select this feeder; the generator validates/skips it below.
  if [[ -d "${PRIOR_CONTROL}/${corpus}/${store_name}" ]]; then
    mkdir -p "${STORE_ROOT}/${corpus}"
    cp -a "${PRIOR_CONTROL}/${corpus}/${store_name}" "${STORE_ROOT}/${corpus}/${store_name}"
  fi
done < <(
  python3 - "${ORIGINAL_JSON}" <<'PY'
import json, os, sys
p = json.load(open(sys.argv[1]))
seen = set()
for audit in p["audits"]:
    meta = audit["meta"]
    seen.add((meta["corpus"], os.path.basename(meta["feeder"])))
order = {name: i for i, name in enumerate(("SMART-DS_1000", "new_dss_data", "dss_data", "minimal_component"))}
counts = {}
for corpus, store in sorted(seen, key=lambda x: (order[x[0]], x[1])):
    counts[corpus] = counts.get(corpus, 0) + 1
    print(f"{corpus}\t{store}")
if any(counts.get(corpus) != 3 for corpus in order):
    raise SystemExit(f"expected three eligible feeders per corpus, got {counts}")
PY
)

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
  --resume \
  --validate-stores \
  --fail-on-skipped-feeders \
  --strict-exit

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
"${PY}" -u "${AUDIT_ROOT}/gridfm/probes/y_recovery_audit.py" \
  --root "${STORE_ROOT}" \
  --corpora "${CORPORA[@]}" \
  --per-corpus 3 \
  --k 1 2 4 8 16 32 \
  --holdout 39 \
  --output "${RUN_ROOT}/fixed_y_audit.json"
