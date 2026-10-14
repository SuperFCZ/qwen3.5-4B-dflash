#!/usr/bin/env bash
# These are human-supplied historical paths, not guessed generated OPP directories.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
base=/home/w00949577/qwen35-dflash-run
export AC_OUTPUT_DIR="${AC_OUTPUT_DIR:-$base/ac-$(date +%Y%m%d-%H%M%S)}"
exec bash "$here/run_server.sh" \
  --a51-suite "$base/a51-fallback-20261012-063342/suite.json" \
  --c-release "$base/head-release-20261014-020633/release.json" \
  --head-capture "$base/draft-head-UCt9vR8q/capture/manifest.json" \
  --head-native "$base/head-native-gpJaP6vN/models/native.json" \
  --native-deployment "$base/w8-nz-full-tsUMDuhH/bundle/deployment-manifest-w8a16.json" \
  --replay16 "$base/debug-draft-inorenr2/private.json" \
  --replay64 "$base/debug-draft-eoutixoi/private.json" \
  --a16-bundle "$base/draft-quant-a4-20261008-015136/c16/capture/manifest.json" \
  --a64-bundle "$base/draft-quant-a4-20261008-015136/c64/capture/manifest.json" "$@"
