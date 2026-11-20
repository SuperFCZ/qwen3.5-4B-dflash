"""This function is embedded verbatim in an isolated ATC wrapper. It also runs in CPU tests."""


def select_scale_source(inputs, outputs, kernel_name, normal_source, rules, package):
    import hashlib
    import os
    from pathlib import Path

    def check(ok, message):
        if not ok:
            raise ValueError("Gate scale ATC dispatch: " + message)

    check(len(inputs) == 3 and len(outputs) == 1, "operator arity")
    x, w, s = inputs
    y = outputs[0]
    for desc, dtype, fmt in (
        (x, "float16", "ND"),
        (w, "int8", "FRACTAL_NZ"),
        (s, "float16", "ND"),
        (y, "float16", "ND"),
    ):
        check(isinstance(desc, dict) and desc.get("dtype") == dtype and desc.get("format") == fmt, "ABI")
    possible = set(rules["shapes"])
    for desc, axes in ((x, (0, 1)), (s, (1, 2)), (y, (0, 2))):
        for key in ("shape", "ori_shape"):
            dims = desc.get(key)
            if not isinstance(dims, (tuple, list)) or len(dims) != 2:
                continue
            for index, (axis, dim) in enumerate(zip(axes, dims)):
                if type(dim) is int and dim > 0:
                    bound = dim * 128 if desc is s and index == 0 else dim
                    possible = {tag for tag in possible if int(tag.split("x")[axis]) == bound}
    # Static GE may compile through a symbolic descriptor. Bind its actual compiler key
    # to the selected ELF symbols of the frozen full graph, including any shared key.
    key = kernel_name.split("__kernel")[0]
    known = rules["keys"].get(key)
    if known is not None:
        possible &= set(known)
    check(bool(possible), "contradictory or unsupported shape/key " + str(kernel_name))
    choices = {"transpose" if tag == "16x2560x19456" else "brcb" for tag in possible}
    check(len(choices) == 1, "ambiguous dynamic Gate/non-Gate dispatch " + str(kernel_name))
    choice = choices.pop()
    check(not os.environ.get("BUILD_KERNEL_SRC"), "source override")
    raw = Path(normal_source).resolve(strict=True)
    check(str(raw) == package["brcb"]["path"], "unbound original SDK source")
    spec = package[choice]
    path = Path(spec["path"])
    check(
        path.resolve(strict=True) == path and hashlib.sha256(path.read_bytes()).hexdigest() == spec["sha256"],
        "selected translation unit identity",
    )
    check(
        hashlib.sha256((path.parent / package["header_name"]).read_bytes()).hexdigest()
        == package["header_sha256"],
        "selected header identity",
    )
    work = Path(os.environ["AI_RUN_DIR"])
    check(
        work.is_absolute()
        and work.resolve(strict=True) == work
        and work / "compiler-cache" == Path(os.environ["ASCEND_CACHE_PATH"]),
        "wrong ATC work/cache",
    )
    record = dict(kernel_name=kernel_name, matching_shapes=sorted(possible), choice=choice, source=spec)
    return str(path), (work, record)


def record_scale_compile(receipt):
    """Journal a route only after the original SDK compile_op returned normally."""
    import fcntl
    import json

    work, record = receipt
    with (work / "scale-dispatch.jsonl").open("a", encoding="utf8") as log:
        fcntl.flock(log, fcntl.LOCK_EX)
        log.write(json.dumps(record, sort_keys=True) + "\n")
        log.flush()
