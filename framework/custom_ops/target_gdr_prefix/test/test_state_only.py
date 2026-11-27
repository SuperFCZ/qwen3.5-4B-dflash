from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gdr_package import FIELDS, VECTOR_WORK, READOUT_MM, helper_barriers, specialize, generate_kernel
from gdr_graph import replace_commit, COMMIT
from test_gdr_prefix import model


def test_existing_event_handoffs_cannot_be_erased():
    stuff = dict(
        balanced="SetFlag<HardEvent::M_V>(0); WaitFlag<HardEvent::M_V>(0); PipeBarrier<PIPE_V>();",
        external="WaitFlag<HardEvent::MTE3_MTE1>(Event_ID);",
        nested="external();",
        conditional="if (load) SetFlag<HardEvent::M_V>(0); WaitFlag<HardEvent::M_V>(0);",
    )
    assert helper_barriers("balanced", stuff) == ["PIPE_V"]
    for k in ("external", "nested", "conditional"):
        assert helper_barriers(k, stuff) is None


def source_fixture():
    # Dummy method shapes and an incoming event reproduce the source editor's control-flow contract.
    a = [""] * 1903

    def put(i, s):
        a[i - 1] = s

    put(1, '#include "kernel_operator.h"')
    put(2, '#include "lib/matmul_intf.h"')
    put(58, "__aicore__ inline void MatMul(int a);")
    put(126, "void Fixture::Init() {")
    put(127, " + ".join("tiling->" + k for k in FIELDS) + ";")
    put(168, "InitWorkSpace(workspace);")
    put(169, "}")
    put(297, "void Fixture::StoreAttnHalfWithTail() { Store(); }")
    put(309, "void Fixture::Transpose_64_128() { PipeBarrier<PIPE_V>(); }")
    put(385, "void Fixture::L2normDim128FloatTrans() { Transpose_64_128(); }")
    put(489, "void Fixture::UB2L1_ND2NZ() { WaitFlag<HardEvent::MTE2_MTE3>(1); }")
    put(523, "void Fixture::L0C2UB_NZ2ND() { DataCopy(); }")
    put(538, "__aicore__ inline void ChunkGatedDeltaRuleKernel<Mode>::MatMul(int a) {")
    put(546, "if (load) WaitFlag<HardEvent::MTE3_MTE1>(Event_ID);")
    put(554, "SetFlag<HardEvent::MTE1_M>(0);")
    put(555, "WaitFlag<HardEvent::MTE1_M>(0);")
    put(560, "Mmad(dst, a, b, params);")
    put(562, "SetFlag<HardEvent::M_V>(0);")
    put(563, "WaitFlag<HardEvent::M_V>(0);")
    put(564, "L0C2UB_NZ2ND(dst, tmp);")
    put(565, "}")
    put(1199, "void Fixture::Process() {")
    put(1296, "Duplicate(ub_temp2, 0.0f, chunkSize*chunkSize);")
    put(1309, "for (int l=1;l<chunkSize;l++) {")
    put(1310, "Exp(ub_temp2[l*chunkSize], ub_temp4[l*chunkSize], l);")
    put(1311, "}")
    for i, op in VECTOR_WORK.items():
        put(i, op + "(x); // placeholder")
    for i in (1315, 1385, 1442, 1486, 1505, 1525, 1531, 1564, 1566):
        put(i, "MatMul<1, true, false>(x);")
    put(1545, "SetFlag<HardEvent::MTE2_V>(0);")
    put(1580, "StoreState();")
    put(1582, "}")
    put(1892, 'extern "C" __global__ __aicore__ void chunk_gated_delta_rule(int p) {')
    put(1893, "GET_TILING_DATA(t,p);")
    put(1900, "}")
    put(1901, "// suffix")
    return "\n".join(a)


def test_edit_preserves_incoming_wait_l0_loads_and_state_matmuls():
    src = source_fixture()
    edited, proof = specialize(src, "NEW_PUBLIC_ENTRY")
    assert "WaitFlag<HardEvent::MTE3_MTE1>(Event_ID)" in edited
    assert "if (prefixCompute) { Mmad(" in edited and "MatMul(bool prefixCompute, int a)" in edited
    assert sorted(READOUT_MM) == proof["guarded_cube_calls"]
    assert edited.count("MatMul<1, true, false>(true, x)") == 6
    assert edited.count("MatMul<1, true, false>(tiling->pruneLevel < 1, x)") == 2
    assert "StoreState();" in edited and "NEW_PUBLIC_ENTRY" in edited and "// suffix" in edited
    assert "l < seqLen || l + 1 == chunkSize" in edited
    assert "chunk_gated_delta_rule(" not in edited
    assert set(proof["kept_event_helpers"]) == {1375, 1508, 1518}
    with pytest.raises(ValueError):
        generate_kernel(src.encode(), "ENTRY")


def test_full_graph_substitutes_only_published_state():
    m, abi = model()
    graph = m.graph[0]
    first = next(n for n in graph.op if n.name == "first").SerializeToString(deterministic=True)
    result = replace_commit(m, abi, level=1)
    assert result["t0_recurrent"]["new"] == "target_gdr_prefix_t0_recurrent"
    nodes = {n.name: n for n in graph.op}
    assert nodes["first"].SerializeToString(deterministic=True) == first
    assert list(nodes["net"].input) == ["arg7:0", "target_gdr_prefix_t0_recurrent:0", "first:1"]
    new = nodes[result["t0_recurrent"]["new"]]
    assert new.type == COMMIT and new.attr["prune_level"].i == 1
    assert list(new.input) == ["arg1:0", "arg2:0", "arg3:0", "arg4:0", "arg5:0", "commit_rows:0"]
