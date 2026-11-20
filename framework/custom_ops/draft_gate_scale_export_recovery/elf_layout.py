"""Bounded ELF64 little-endian reader shared by native and AscendC image diagnostics."""

import hashlib
import struct

from archive import require

EH = struct.Struct("<16sHHIQQQIHHHHHH")
PH = struct.Struct("<IIQQQQQQ")
SH = struct.Struct("<IIQQQQIIQQ")
SYM = struct.Struct("<IBBHQQ")


def table(blob):
    require(len(blob) >= EH.size, "truncated ELF header")
    h = EH.unpack_from(blob)
    require(
        h[0][:7] == b"\x7fELF\x02\x01\x01"
        and h[3] == 1
        and h[8] == EH.size
        and h[11] == SH.size
        and 0 < h[13] < h[12]
        and h[6] + h[12] * SH.size == len(blob),
        "unsupported ELF64 section table",
    )
    rows = [SH.unpack_from(blob, h[6] + i * SH.size) for i in range(h[12])]
    n = rows[h[13]]
    require(n[1] == 3 and n[4] + n[5] <= len(blob), "invalid section-name string table")
    strings = blob[n[4] : n[4] + n[5]]
    results = []
    for idx, (nm, typ, flags, va, off, size, link, info, align, ent) in enumerate(rows):
        end = strings.find(b"\0", nm)
        require(nm <= end and (typ == 8 or off + size <= len(blob)), "ELF section bounds")
        results.append(
            dict(
                index=idx,
                name=strings[nm:end].decode("ascii"),
                type=typ,
                flags=flags,
                va=va,
                offset=off,
                bytes=size,
                link=link,
                info=info,
                alignment=align,
                entry_size=ent,
                sha256=None if typ == 8 else hashlib.sha256(blob[off : off + size]).hexdigest(),
            )
        )
    return h, results
