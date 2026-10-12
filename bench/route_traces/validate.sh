#!/usr/bin/env bash
# Validates every recorded trace with host_tier.py and captures the verdicts for the manifest.
#
#   bash bench/route_traces/validate.sh
#
# Writes out/route-traces/validate.txt with one "=== <name>" section per trace.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=out/route-traces
: > "$OUT/validate.txt"
for trace in "$OUT"/*.bin; do
  name="$(basename "$trace" .bin)"
  echo "=== $name" | tee -a "$OUT/validate.txt"
  python3 tools/expert_cache_replay/host_tier.py "$trace" --validate 2>&1 | tee -a "$OUT/validate.txt"
done
