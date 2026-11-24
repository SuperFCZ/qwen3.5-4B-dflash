"""The six measured W8A16 implementations for the real five-layer checkpoint."""

from dataclasses import dataclass
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
sys.path[:0] = [str(OPS / "draft_kv_bandwidth"), str(OPS / "draft_full_panels")]
import kv_panel as kv
import o_panel as o
import q_panel as q
import fc_panel as fc
import packed_panel as packed
from joint import c, down, gate, require, OLD_OP

FAMILIES = ("fc", "kv", "q", "o", "gate_up", "down")
FULL = "full-draft-panels-UFqlys"
LATENCY = "5d5fec70a1b842c2bd71144bb80746ce0cc6fb40e2668872ebc14446e27efe42"
FULL_OM = {
    16: "8ba2d07ac3a53bd639d5a0ebed170ef7784286b2e1f540b36507fff78ed95d7f",
    64: "85ec46551789ba032f88fb66c7fed49731733e603e262d0338ab35e76aea25ab",
}


@dataclass(frozen=True)
class Projection:
    op: str
    stem: str
    mode: str
    m: int
    k: int
    n: int
    columns: int
    panel: int
    pack_x: bool = False

    @property
    def shape(self):
        return self.m, self.k, self.n


def selections(gear):
    require(gear in (16, 64), "unknown Draft gear")
    small = gear == 16
    return dict(
        gate_up=Projection(gate.OP, gate.STEM, "panel-k2560-w2", 16, 2560, 19456, 64, 2560),
        down=Projection(down.OP, down.STEM, down.SINGLE, 16, 9728, 2560, 64, 4864),
        q=Projection(q.OP, q.STEM, "q-k2560-w2-next-n", 16, 2560, 4096, 64, 2560),
        o=Projection(o.OP, o.STEM, o.PANEL, 16, 4096, 2560, 64, 4096),
        kv=Projection(
            kv.OP,
            kv.STEM,
            kv.DIRECT if small else "kv-n128-k1280",
            gear + 16,
            2560,
            2048,
            64 if small else 128,
            2560 if small else 1280,
        ),
        fc=Projection(
            fc.OP if small else packed.OP,
            fc.STEM if small else packed.STEM,
            fc.LOCAL if small else packed.MODES[160],
            gear,
            12800,
            2560,
            64 if small else 160,
            2560 if small else 1280,
            not small,
        ),
    )


def sites(specs):
    return {"fc": specs["fc"], **{f"layer{i}.{kind}": specs[kind] for i in range(5) for kind in FAMILIES[1:]}}


@dataclass(frozen=True)
class Receipt:
    root: str
    result: str
    build: str
    atc: str
    pair: str
    om: str
    text: str


Q = Receipt(
    "q-bandwidth-zygJUl",
    "2dd49d503d1f9d087b9737dfcadc130d0b13f92171ef00eb4e02a7465f8500bc",
    "build-next-n",
    "atc-next-n",
    "benchmark",
    "ed321a80377318dba4804cc7deafb6698386470ad7a2d09d5069f2394249403b",
    "3cf76e1de98a29cdfe87d0c007fe455204ac8e1cafea469e9e51434832ff0495",
)
O = Receipt(
    "o-bandwidth-F1HnBO",
    "fadaf4ab34667b0cdb9648ce7a9e04666e338de27cabfc2c5f24172d62bf7321",
    "build-panel",
    "atc-" + o.PANEL,
    "benchmark",
    "d0d081acd7dbf932b00d5f321722a89d053aeba9f67dba81125f51f7e485255f",
    "43fbebe602f4aa0e4fefd7d80aa3e3189fd9f3ec5bfd59e99f45e9e0630fe013",
)
SINGLE = {
    16: dict(
        q=Q,
        o=O,
        fc=Receipt(
            "fc-bandwidth-6bPlhz",
            "90eef9b84589579c3c52b3b5685c24e45657cdd4a3e331b0ae5215ec05e00902",
            "build-" + fc.LOCAL,
            "atc-c16-" + fc.LOCAL,
            "benchmark-c16",
            "d03092dd04a0b3b6bed4d76836ce47efdb8ae9146469363b029d589734facb6d",
            "a3c062d477d371d19e0549eb2ac1eafe8a61a25278c399360c4776fdd551f1ca",
        ),
        kv=Receipt(
            "kv-bandwidth-a6xFlL",
            "b817354039de6e48cc637b5904012aebb0445ee783d2b2493887ad08020ef558",
            "build-" + kv.DIRECT,
            "atc-c16-" + kv.DIRECT,
            "benchmark-c16-raw-nz-brcb__" + kv.DIRECT,
            "877402d08a5753550cc427179096f85a01a01ac4bb1c6bdb1e19a1deea00da64",
            "5810be462a109557ab8b99362bff3cc2709cb9fd07f7552db80821a9f9a22a35",
        ),
    ),
    64: dict(
        q=Q,
        o=O,
        fc=Receipt(
            "fc-bandwidth-Chqsh4",
            "57a6c2b9c0676699021324625b13e32a880194aabac0a39cc1d6e6b5fac50125",
            "build-fc-pack-a-n160",
            "atc-fc-pack-a-n160",
            "benchmark-fc-pack-a-n64__fc-pack-a-n160",
            "e9d43f4c893384576f2427638e6ae6b6d4ee463f84c238c3623159aa8ece9499",
            "efe620b4a1aa74b6729a7417cdbd2aeb57621d4972d6b99ef9af20087d3c1df6",
        ),
        kv=Receipt(
            "kv-bandwidth-nafZjO",
            "c714b45271e48226693820ef423982c840555d15120f737130d08a2f54c86314",
            "build-kv-n128-k1280",
            "atc-c64-kv-n128-k1280",
            "benchmark-c64-raw-nz-brcb__kv-n128-k1280",
            "40e6b9a5281f7476f441192b32e9593e68850c33f7c7487e01116881ee5525a9",
            "559dad0cf7e25b9f3c13844f3e3f29fee9735fd1463b01c2b792afa176d93701",
        ),
    ),
}
