# Offline validation

No server execution was performed by the developer. No single Gate or Full Draft OM was compiled locally with ATC or loaded/executed. There are no local speedups or device Golden results.

In Docker `qwen35-cann9-host`, Linux ARM64, CANN root /usr/local/Ascend/cann-9.0.0, using the prior single-operator general-shape Scheduler A compile fixture:

- `/work/build/gate8-check-t6eu9_qo/report.json`: **LOCAL_GATE8_RUNNER_BOTH_HOST_DEVICE_BUILDS_AND_SERIALIZER_MATCH**. The preceding `gate8-check-mcofvqs7` build had the same Off/Raw ELF and Runner digests.
- Built the new C++ ACL OM Runner; ELF SHA256 `374bcb8759e36bc4bef349c441d0e6761ef16dd7149ad4bab7e928871dc5281c`, 249912 B. Local ldd resolved. This is a compile/link check, not a real Driver/Runtime admission.
- Fresh Off and Raw CPU tiling/proto DSOs and Gate machine-code builds with the generated CANN CMake target `DFlashGroupQuantLinear_ascend310p`; same math source, differing generated Gate flag. No Head device target.
- Off package target ELF: 57216 B, SHA256 `831a9ce01371ef0a70962e6576489f8d16ec07daa253b99f1b2e5e627f71bca3`. Matches the file SHA reported for the accepted server control. Executable section: 52300 B, SHA256 `df644d4bf4d7400b33399105415f667502ae927a515955daf3217937d6a5df34`.
- Raw package target ELF: 57216 B, SHA256 `b84883a22351196622fc922355b2ca7b92548b721c1d9fa43fa7faeea31b35bc`. Executable section: 52360 B, SHA256 `26a57a1045f65f4809fc950f8f1cee952a1aae2f4397dab693204510bb380361`. Equal file length does not imply equal code. This Raw binary is **LOCAL_ONLY_NOT_SERVER_VERIFIED**. Neither package target is proof of the actual code emitted by ATC in a new OM.
- Off/Raw generated Gate dispatch configuration and registry entry compared equal. Single-op local config SHA `a2395d2553df87ad1ce710e6f8c1019f83fae7dfa63a5020d17a92ccce3ad463`; single-op index `13907438319d6fad319a240e3d038be38f7710adc119d17009b10545d77f9036`. The real server joint package must validate its own entries.
- Ran the real SDK serializer on CPU, not a TilingContext mock. Marker status **GATE8_PARENT_224_PREFIX_24_CUBE_200_FIELDS_50_MATCH**. The server performs its own serializer check and actual ATC callback verification.
- CPU contract suite: 16 tests, including a simulated complete four-model smoke and three-round full timing replay. They use synthetic reports/graphs/ELFs/capture data only. Negative coverage includes graph/source substitution, incorrect blockDim or dispatch, different executable bytes/metadata, field and resource drift, swapped/omitted layers, poisoned/incorrect result, safety failures, wrong DSO, predecessor mixing, forward/reverse order, P95 regression, <1.03, and denying optional integration. The pre-existing 21 raw-prefetch/layer CPU checks also passed on the development host.

Actual five witnessed single AIR/OM sources, new model snapshots and the full server OPP/mount namespace exist only on the server. Source/Build/Export, both gears' device Smoke and formal timing, Compare, and optional full stages are **NOT_RUN** for the new server experiment. No SDK source, frozen tensor or generated binary is included in this commit.
