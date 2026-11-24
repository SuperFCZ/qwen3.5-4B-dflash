"""Bind Layer0 context and proposal rows and both FP16 output halves to the accepted full Draft."""

import numpy as np
from kv_panel import ROLE, c, require, shape
from o_source import read_projection_source

NZ = "0ff98c839093ce63e5363a88406d359fe8a8992dc3f877f5d5f83b68b7964d67"
GN = "dab7d3d32d72616646636010ee7dfab7ec7a851e279101a87ca697f93477358c"
PINS = {
    16: (
        "b559fd8cf524eeb92d93fa2e9cb4a8df29601958392110b83cf5264932919914",
        14604,
        "6952e057c1a5e820a4a5f78f5b9648926e5e055bddc1ef3b61c43894b17ed879",
    ),
    64: (
        "e5aa923ff7d6a7cee7b41de2aa57bdc38732c2a51a40860b82111732951f8acd",
        14344,
        "18a556ceaeec809f7dfdde252c01df723f2ec51d4dca84f95d95e8341097eb50",
    ),
}


def read(out, root, types, gear):
    src = read_projection_source(
        out, root, types, gear, ROLE, ("fc", ROLE), shape(gear), (NZ, GN, *PINS[gear])
    )
    require(src["kv_node"] == "proj_layer0_kv", "wrong Layer0 KV node")
    src["capture_origin"] = "LAYER0_FC_NORM_EMBEDDING_KV_DEVICE_PREFIX"
    # Change both sides of the context/proposal boundary for the independent dense stress.
    replay = src["replays"]
    public = c.verify(replay["measured"]["features"])
    features = np.fromfile(public, "<f2").reshape(1, gear, 12800)
    other = np.roll(features[:, ::-1], 129, axis=2).copy()
    other.view("<u2")[:, :, ::2] ^= 0x8000
    file = out / "stress-features.bin"
    other.tofile(file)
    replay["stress"]["features"] = c.identity(file)
    require(
        replay["stress"]["features"]["sha256"] != replay["measured"]["features"]["sha256"],
        "stress input equals measured context",
    )
    return src
