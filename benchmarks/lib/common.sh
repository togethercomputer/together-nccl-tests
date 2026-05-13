#!/usr/bin/env bash
# lib/common.sh — Shared definitions for NCCL benchmark scripts
#
# Source this from run_k8s.sh / run_slurm.sh / run_mpi.sh:
#   source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

# ── Registry credentials ──────────────────────────────────────────────────────
# Set DOCKER_USER / DOCKER_TOKEN in the environment, or place them in a
# .env file at the repository root (automatically loaded; never committed):
#
#   DOCKER_USER=johnsontogether
#   DOCKER_TOKEN=dckr_pat_xxxx
#   DOCKER_REGISTRY=docker.io   # optional, default: docker.io
#
# Variables already in the environment take precedence over .env.

_load_dotenv() {
    local dotenv
    dotenv="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
    [[ ! -f "$dotenv" ]] && return 0
    local line key value
    while IFS= read -r line || [[ -n "$line" ]]; do
        # skip blank lines and comments
        [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
        key="${line%%=*}"; key="${key// /}"
        value="${line#*=}"
        # only process DOCKER_* keys; don't overwrite existing env vars
        [[ "$key" != DOCKER_* ]] && continue
        [[ -n "${!key:-}" ]] && continue
        # strip surrounding quotes
        value="${value#\"}"; value="${value%\"}"
        value="${value#\'}"; value="${value%\'}"
        export "${key}=${value}"
    done < "$dotenv"
}
_load_dotenv

# Print a reminder when credentials are missing.
# $1 = context string shown in the message (e.g. "K8s imagePullSecret")
_warn_no_creds() {
    local context="${1:-image pull}"
    local repo_root
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    echo ""
    echo "  WARNING: DOCKER_USER / DOCKER_TOKEN not set — ${context} may fail."
    echo ""
    echo "  Option 1 — .env file (persists across sessions):"
    echo "    cp ${repo_root}/.env.example ${repo_root}/.env"
    echo "    # then edit .env:"
    echo "    #   DOCKER_USER=johnsontogether"
    echo "    #   DOCKER_TOKEN=dckr_pat_xxxx"
    echo ""
    echo "  Option 2 — export in your shell:"
    echo "    export DOCKER_USER=johnsontogether"
    echo "    export DOCKER_TOKEN=dckr_pat_xxxx"
    echo ""
}

# Create or update the K8s imagePullSecret from DOCKER_USER/TOKEN.
# Warns and returns if credentials are not set.
registry_auth_k8s() {
    local namespace="${1:-default}"
    local secret_name="${2:-dockerhub-nccl-tests}"
    local registry="${DOCKER_REGISTRY:-docker.io}"
    if [[ -z "${DOCKER_USER:-}" || -z "${DOCKER_TOKEN:-}" ]]; then
        _warn_no_creds "K8s imagePullSecret '${secret_name}'"
        return 0
    fi
    echo "  Syncing imagePullSecret '${secret_name}' (${DOCKER_USER}@${registry})..."
    kubectl create secret docker-registry "${secret_name}" \
        --docker-server="${registry}" \
        --docker-username="${DOCKER_USER}" \
        --docker-password="${DOCKER_TOKEN}" \
        --namespace="${namespace}" \
        --dry-run=client -o yaml | kubectl apply -f - >/dev/null
    echo "  Secret synced."
}

# docker login on the local host. Warns and returns if credentials are not set.
registry_auth_local() {
    local registry="${DOCKER_REGISTRY:-docker.io}"
    if [[ -z "${DOCKER_USER:-}" || -z "${DOCKER_TOKEN:-}" ]]; then
        _warn_no_creds "docker login on local host"
        return 0
    fi
    echo "  docker login ${registry} as ${DOCKER_USER}..."
    echo "${DOCKER_TOKEN}" | docker login "${registry}" \
        -u "${DOCKER_USER}" --password-stdin
}

# docker login on every host in a hostfile (token via SSH stdin, not args).
# Warns and returns if credentials are not set.
registry_auth_remote() {
    local hostfile="${1:-}"
    local registry="${DOCKER_REGISTRY:-docker.io}"
    [[ -z "$hostfile" || ! -f "$hostfile" ]] && return 0
    if [[ -z "${DOCKER_USER:-}" || -z "${DOCKER_TOKEN:-}" ]]; then
        _warn_no_creds "docker login on remote hosts"
        return 0
    fi
    local host
    while IFS= read -r line || [[ -n "$line" ]]; do
        host=$(echo "$line" | awk '{print $1}')
        [[ -z "$host" || "$host" == \#* ]] && continue
        echo "  docker login on ${host}..."
        echo "${DOCKER_TOKEN}" | ssh "$host" \
            "docker login ${registry} -u '${DOCKER_USER}' --password-stdin" \
            && echo "    OK" || echo "    WARN: login failed on ${host}"
    done < "$hostfile"
}

# ── Test matrix ───────────────────────────────────────────────────────────────
# Format: "test_name:label:NCCL_ALGO:NCCL_NVLS_ENABLE:NCCL_COLLNET_ENABLE"
# NCCL_ALGO empty = do not set (let NCCL auto-select; scripts should also
# unset NCCL_ALGO in case the cluster sets it globally via /etc/environment)
NCCL_TEST_MATRIX=(
    "all_reduce:ring:RING:0:0"
    "all_reduce:nvls::1:0"
    "all_reduce:collnet_sharp::0:1"
    "all_gather:ring:RING:0:0"
    "all_gather:nvls::1:0"
    "all_gather:collnet_sharp::0:1"
    "reduce_scatter:ring:RING:0:0"
    "reduce_scatter:nvls::1:0"
    "reduce_scatter:collnet_sharp::0:1"
)

# Filter matrix to a single test type via NCCL_TESTS env var (e.g. NCCL_TESTS=all_reduce)
if [[ -n "${NCCL_TESTS:-}" ]]; then
    _filtered=()
    for _entry in "${NCCL_TEST_MATRIX[@]}"; do
        [[ "${_entry%%:*}" == "$NCCL_TESTS" ]] && _filtered+=("$_entry")
    done
    NCCL_TEST_MATRIX=("${_filtered[@]}")
    unset _filtered _entry
fi

# ── GPU type detection ────────────────────────────────────────────────────────
detect_gpu_type() {
    local name="" raw
    # Only use nvidia-smi output on successful exit; a failed probe (e.g. on a
    # login node) can write error text to stdout which would produce garbage paths.
    if raw=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null); then
        name=$(echo "$raw" | awk 'NR==1{print; exit}')
    fi
    if   [[ "$name" == *"GB200"* ]]; then echo "GB200"
    elif [[ "$name" == *"B200"*  ]]; then echo "B200"
    elif [[ "$name" == *"H200"*  ]]; then echo "H200"
    elif [[ "$name" == *"H100"*  ]]; then echo "H100"
    elif [[ "$name" == *"A100"*  ]]; then echo "A100"
    elif [[ -n "$name" ]]; then echo "$(echo "$name" | tr ' ' '_')"
    else echo "unknown"
    fi
}

# ── Extract peak busBW from a result file ────────────────────────────────────
peak_busbw() {
    grep -E "^[[:space:]]+[0-9]" "$1" 2>/dev/null \
        | tail -1 | awk '{print $8}' || echo "N/A"
}

# ── Print summary table with optional baseline comparison ────────────────────
print_summary() {
    local results_dir="$1"
    local baseline_dir="${2:-}"

    echo ""
    echo "============================================================"
    echo "SUMMARY — Peak busBW at largest message size"
    echo "============================================================"

    local has_baseline=0
    if [[ -n "$baseline_dir" && -d "$baseline_dir" ]]; then
        has_baseline=1
        printf "%-22s %-18s %14s %14s %10s %8s\n" \
            "Test" "Config" "Baseline(GB/s)" "Current(GB/s)" "Delta(GB/s)" "Delta(%)"
        printf "%-22s %-18s %14s %14s %10s %8s\n" \
            "----" "------" "-------------" "------------" "----------" "--------"
    else
        printf "%-22s %-18s %14s\n" "Test" "Config" "busBW (GB/s)"
        printf "%-22s %-18s %14s\n" "----" "------" "------------"
    fi

    local outfile fname test_name label current
    for outfile in $(ls "${results_dir}"/*.out 2>/dev/null | sort); do
        [[ -f "$outfile" ]] || continue
        fname=$(basename "$outfile" .out)
        # Split on the known test-name prefix so that labels containing
        # underscores (e.g. collnet_sharp) are not truncated.
        test_name=""
        label=""
        for _prefix in "all_reduce" "all_gather" "reduce_scatter"; do
            if [[ "$fname" == "${_prefix}_"* ]]; then
                test_name="$_prefix"
                label="${fname#${_prefix}_}"
                break
            fi
        done
        [[ -z "$test_name" ]] && { test_name="$fname"; label="unknown"; }
        current=$(peak_busbw "$outfile")
        [[ -z "$current" ]] && current="N/A"

        if [[ $has_baseline -eq 1 ]]; then
            local baseline="N/A" delta="N/A" pct="N/A"
            local baseline_file="${baseline_dir}/${fname}.out"
            if [[ -f "$baseline_file" ]]; then
                baseline=$(peak_busbw "$baseline_file")
            fi
            if [[ "$baseline" != "N/A" && "$current" != "N/A" \
                  && -n "$baseline" && -n "$current" ]]; then
                delta=$(awk "BEGIN {printf \"%.2f\", ${current} - ${baseline}}")
                pct=$(awk  "BEGIN {printf \"%.1f\", (${current} - ${baseline}) / ${baseline} * 100}")
                [[ $(awk "BEGIN {print (${delta} >= 0)}") -eq 1 ]] \
                    && delta="+${delta}" && pct="+${pct}"
            fi
            printf "%-22s %-18s %14s %14s %10s %8s\n" \
                "$test_name" "$label" "$baseline" "$current" "$delta" "${pct}%"
        else
            printf "%-22s %-18s %14s\n" "$test_name" "$label" "$current"
        fi
    done
}
