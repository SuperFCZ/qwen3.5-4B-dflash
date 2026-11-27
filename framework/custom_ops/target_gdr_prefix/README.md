# Target GDR prefix commitment on Ascend310P3

The deployed Chunk Verify first evaluates the proposal block. After Target acceptance is known,
each of 24 GDN layers starts from its FP32 round-start state and commits `accepted_count + 1`
rows. The compiler-visible state is always `[1,32,128,128]`. Query contributes to attention
output. The second pass publishes only its final recurrent state.

Run on the receiver with the four-column convolution Verify baseline.

```bash
bash framework/custom_ops/target_gdr_prefix/run.sh --from-run target-conv-window-qyWq56
bash framework/custom_ops/target_gdr_prefix/state_only.sh --from-probe target-gdr-prefix-RUN
```

The first command profiles both native passes and freezes all 16 accepted-prefix lengths
from the first layer's genuine operands. It also checks the native token-recurrent route
against Chunk. The second command builds a private `TargetGdrPrefixState` from the locally
installed, fingerprinted Chunk arithmetic. The original installation stays read-only.
The new GE operator consumes key, value, gate, beta, the initial state and the device length.
It emits one FP32 state. The first-pass attention and its discard-state buffer remain live.

The clone checks the fixed shape and private workspace handoff. The terminal variant disables
the two final readout products. The readout variant additionally disables the query/key readout
product and its exclusive preparation where the event contract allows. The short-mask variant
evaluates exponential mask rows only for the live prefix and the final decay row used in the
state update. Source generation keeps
the native MatMul input loads, incoming event waits and state products. Its patch manifest records
which helpers were retained. Generated receiver source, binaries, profiles and tensors stay in
the new run directory on the server.

Only variants matching all 524,288 FP32 elements at each length on real and independent stress
inputs enter timing. Selection uses the archived frequency of each length from 1 to 4 and
three paired Execute+Sync rounds. The full receiver profiles representative lengths, checks
all outputs and mutated reference buffers, then runs only the candidate on the original 20
prompts. The historical archive supplies the comparison for latency, throughput, token IDs,
stopping and complete speculative round traces.
