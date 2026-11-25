"""Preregistered single-projection transfers. Each case uses its own measured incumbent."""

from dataclasses import dataclass, replace
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
sys.path[:0] = [str(OPS / "draft_gate_next_n"), str(OPS / "draft_all_w8")]
from selection import c, require, selections, fc, packed, kv, q, o, gate, down, SINGLE, OLD_OP
import gate_next


@dataclass(frozen=True)
class Recipe:
    role: str
    gear: int
    name: str
    width: int
    panel: int
    chunk: int
    base_k: int
    banks: int = 2
    packet: int = 1
    ahead: bool = True
    keep_a: bool = True
    activation: str = "row"  # row/gather expose ND; packed includes the graph transpose

    @property
    def spec(self):
        return selections(self.gear)[self.role]

    @property
    def shape(self):
        return self.spec.shape

    @property
    def ub(self):
        return 8192 + 16 * self.chunk * (self.packet + 2 * self.banks) + 2 * (self.panel // 128) * self.width

    @property
    def l1(self):
        m, k, _ = self.shape
        return 2 * (m * (k if self.keep_a else self.panel) + self.width * self.panel)

    @property
    def layout(self):
        return self.width, self.panel, self.activation == "packed"

    @property
    def cube_geometry(self):
        return self.width, self.panel, self.base_k, self.l1

    @property
    def abi(self):
        m, k, n = self.shape
        return dict(
            inputs=[dict(name="x", dtype="float16", shape=[m, k])],
            outputs=[dict(name="y", dtype="float16", shape=[m, n])],
        )

    def variant(self, name, **changes):
        return replace(self, name=name, **changes)


CONTROL = {
    "down": Recipe("down", 16, down.SINGLE, 64, 4864, 2432, 256, banks=1, ahead=False),
    "gate": Recipe("gate_up", 16, gate_next.NEXT, 64, 2560, 2560, 256, activation="gather"),
    "o": Recipe("o", 16, o.PANEL, 64, 4096, 2048, 256),
    "q": Recipe("q", 16, "q-k2560-w2-next-n", 64, 2560, 2560, 256),
    "fc16": Recipe("fc", 16, fc.LOCAL, 64, 2560, 2560, 256),
    "kv64": Recipe("kv", 64, "kv-n128-k1280", 128, 1280, 1280, 128),
    "kv16": Recipe("kv", 16, kv.DIRECT, 64, 2560, 2560, 256),
    "fc64": Recipe("fc", 64, packed.MODES[160], 160, 1280, 1280, 80, keep_a=False, activation="packed"),
}


def candidates(key):
    b = CONTROL[key]
    v = b.variant
    if key == "down":
        return [
            v("down-next-w1", ahead=True),
            v("down-raw2-w1", packet=2),
            v("down-next-raw2-w1", ahead=True, packet=2),
            v("down-next-w2", ahead=True, banks=2),
            v("down-n160-k2432", width=160, panel=2432, chunk=2432, base_k=64, keep_a=False, ahead=True),
        ]
    if key == "gate":
        return [
            v("gate-next-row", activation="row"),
            *[
                v(f"gate-n128-k{pk}-{a}", width=128, panel=pk, chunk=pk, base_k=128, activation=a)
                for pk in (2560, 1280)
                for a in ("gather", "row")
            ],
            v("gate-next-raw2", packet=2),
        ]
    if key == "q":
        return [v("q-n128-full", width=128, base_k=128), v("q-next-raw2", packet=2)]
    if key == "o":
        wide = v("o-n160-k2048", width=160, panel=2048, base_k=64)
        return [v("o-n16-raw2", packet=2), wide, wide.variant("o-n160-raw2", packet=2)]
    if key == "fc16":
        wide = v("fc16-n160-k1280", width=160, panel=1280, chunk=1280, base_k=80)
        return [wide, v("fc16-raw2", packet=2), wide.variant("fc16-n160-raw2", packet=2)]
    if key == "kv64":
        return [
            v("kv64-n128-packed", activation="packed"),
            v("kv64-n128-raw2", packet=2),
            v("kv64-n128-packed-raw2", activation="packed", packet=2),
        ]
    # A contiguous two-piece DMA differs from the two separate raw-bank prefetch in Q.
    require(key in ("fc64", "kv16"), "unknown projection")
    return [v(key + "-raw2", packet=2)]


def validate(r):
    m, k, n = r.shape
    require(
        n % (8 * r.width) == 0
        and k % r.panel == 0
        and r.panel % r.chunk == 0
        and r.chunk % 128 == 0
        and r.panel % r.base_k == 0
        and r.width % 16 == 0
        and (r.width // 16 * r.panel // r.chunk) % r.packet == 0
        and r.banks in (1, 2)
        and r.packet in (1, 2)
        and r.activation in ("row", "gather", "packed")
        and 0 < r.ub < 262144
        and r.l1 < 1048576,
        "infeasible transfer: " + r.name,
    )


def header(r):
    validate(r)
    m, k, n = r.shape
    return f"""#pragma once
#include <cstdint>
#define DFLASH_KV_INPUT {2 if r.activation=="packed" else 0}
namespace transfer_panel {{
constexpr uint32_t RowsCount={m}, K={k}, N={n}, Columns={r.width}, Group=128;
constexpr uint32_t PanelK={r.panel}, ChunkK={r.chunk}, BaseKValue={r.base_k}, ChunkN=16;
constexpr uint32_t Panels=K/PanelK, Tiles=N/Columns, Groups=PanelK/Group;
constexpr uint32_t Parts=PanelK/ChunkK, ChunksPerStripe=Parts, GroupsPerPart=ChunkK/Group;
constexpr uint32_t Pieces=Columns/16*Parts, Elements=ChunkK*16, Stripe=Elements;
constexpr uint32_t RawBanks=1, W16Banks={r.banks}, RawPieces={r.packet};
constexpr uint32_t CodeBytes=RawPieces*Elements, ScaleBytes=2*Groups*Columns;
constexpr uint32_t WeightBytes=2*W16Banks*Elements, UserBytes=CodeBytes+ScaleBytes+WeightBytes+8192;
constexpr bool StreamA=false;
template<uint32_t M> struct Activation {{
    static constexpr bool Whole={"true" if r.keep_a else "false"};
    static constexpr uint32_t LocalK=Whole ? K : PanelK;
}};
constexpr uint32_t BaseK(uint32_t) {{ return BaseKValue; }}
constexpr uint32_t ResidentL1(uint32_t m) {{ return 2*(m*Activation<RowsCount>::LocalK+Columns*PanelK); }}
static_assert(K%PanelK==0 && PanelK%ChunkK==0 && ChunkK%Group==0 && Pieces%RawPieces==0, "complete groups");
}}
""".encode()
