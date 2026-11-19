#!/usr/bin/env bash
# CPU-only CANN SDK probe with a named platform. No aclInit/SetDevice or ATC.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=${CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}
out=$(mktemp -d "${TMPDIR:-/tmp}/w16-sdk-query-XXXXXX")
echo "OUTPUT=$out"
libs=()
for name in libtiling_api.a libnnopbase.so libexe_graph.so libregister.so libplatform.so libascendcl.so libc_sec.so; do
  test -f "$root/lib64/$name"
  libs+=("$root/lib64/$name")
done
sha256sum "$here/w16_tiling.cpp" "${libs[@]}" > "$out/inputs.sha256"
c++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -I"$root/include" -I"$root/aarch64-linux/asc/include/utils" \
  "$here/w16_tiling.cpp" "${libs[@]}" -ldl -o "$out/query"
ldd "$out/query" > "$out/ldd.txt"
if grep -F 'not found' "$out/ldd.txt"; then exit 1; fi
"$out/query" > "$out/result.json"
cat "$out/result.json"
