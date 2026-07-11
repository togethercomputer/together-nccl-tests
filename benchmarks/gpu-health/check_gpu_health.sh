#!/bin/bash
#
# check_gpu_health.sh — Query GPU clocks, ECC, and throttle status via nvidia-smi
#
# Outputs CSV: node,gpu,gfx_mhz,gfx_max,mem_mhz,mem_max,ecc,ecc_corr,ecc_uncorr,temp_c,power_w,flags
#
# Usage:
#   Local:   ./check_gpu_health.sh
#   Cluster: srun -N<nodes> --ntasks-per-node=1 bash check_gpu_health.sh
#
set -euo pipefail

HOSTNAME=$(hostname -s)
# Strip common prefixes
HOSTNAME="${HOSTNAME#use3a-ss-b200-}"
HOSTNAME="${HOSTNAME%.cloud.together.ai}"

nvidia-smi --query-gpu=index,clocks.current.graphics,clocks.max.graphics,clocks.current.memory,clocks.max.memory,ecc.mode.current,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total,temperature.gpu,power.draw,clocks_event_reasons.hw_slowdown,clocks_event_reasons.sw_thermal_slowdown \
    --format=csv,noheader,nounits \
| while IFS=', ' read -r idx gc gmax mc mmax ecc corr uncorr temp power hw_slow sw_therm; do
    flags=""
    if [ "$gc" != "$gmax" ]; then flags="CLOCK_LOW"; fi
    if [ "$ecc" != "Enabled" ]; then flags="${flags:+$flags|}ECC_OFF"; fi
    if [ "$uncorr" != "0" ] && [ "$uncorr" != "N/A" ]; then flags="${flags:+$flags|}ECC_UNCORR"; fi
    if [ "$hw_slow" = "Active" ]; then flags="${flags:+$flags|}HW_THROTTLE"; fi
    if [ "$sw_therm" = "Active" ]; then flags="${flags:+$flags|}THERMAL"; fi
    if [ -z "$flags" ]; then flags="OK"; fi
    echo "${HOSTNAME},GPU${idx},${gc},${gmax},${mc},${mmax},${ecc},${corr},${uncorr},${temp},${power},${flags}"
done
