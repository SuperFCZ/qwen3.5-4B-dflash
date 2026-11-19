"""CPU model for a single N tile. Models issue order and worst-case lifetimes, not cycles."""
from collections import Counter


def schedule(k_tiles=10, *, overlap, fail_at=None):
    """Both controls use two W banks, identical DMA order, waits and one fence per K."""
    if k_tiles <= 0 or (fail_at is not None and not 0 <= fail_at < k_tiles):
        raise ValueError("invalid K extent/failure point")
    yield "reserve", -1
    yield "copy", 0
    yield "wait_s", 0
    yield "wait_v", 0
    yield "dequant", 0
    for k in range(k_tiles):
        more = k + 1 < k_tiles
        if more:
            yield "copy", k + 1
        yield "v_to_s", k
        yield ("iterate_false" if k == fail_at else "iterate"), k
        if more:
            yield "wait_s", k + 1
            yield "wait_v", k + 1
        if more and overlap and k != fail_at:
            yield "dequant", k + 1
        yield "all", k
        if k == fail_at:
            break
        if more and not overlap:
            yield "dequant", k + 1
    else:
        yield "get_c", -1
        yield "output_all", -1
    yield "release", -1
    yield "end", -1


def require(condition, description):
    if not condition:
        raise ValueError(description)


def check_trace(events, k_tiles, w_banks=2):
    """An issued B read stays live until a same-core ALL, even after Iterate returns."""
    require(k_tiles > 0 and w_banks in (1, 2), "invalid tile or bank count")
    flags = {}
    ready = set()
    raw = [None, None]
    weights = [None] * w_banks
    raw_readers = set()
    sdk_readers = set()
    copied, converted, accumulated = [], [], []
    handoffs = set()
    fence_due = None
    last_fenced = -1
    failed = published = output_fenced = ended = reserved = released = False
    for op, k in events:
        require(not ended, "work after End")
        if op == "reserve":
            require(not reserved and not released, "double allocation")
            reserved = True
            continue
        if op == "end":
            require(released, "End before drain/release")
            ended = True
            continue
        require(reserved, "resources unavailable")
        if op in ("get_c", "output_all", "release"):
            if op == "get_c":
                require(not failed and not published and last_fenced == k_tiles - 1
                        and accumulated == list(range(k_tiles)), "incomplete/duplicate FP16 output")
                published = True
            elif op == "output_all":
                require(published and not output_fenced, "output fence without output")
                output_fenced = True
            else:
                require(fence_due is None and not flags and not sdk_readers and not raw_readers
                        and (failed or output_fenced), "release with pending work")
                reserved, released = False, True
            continue
        require(0 <= k < k_tiles, "K outside N tile")
        r, w = k % 2, k % w_banks
        if op == "copy":
            require(not failed and k == len(copied) and not flags and r not in raw_readers,
                    "raw overwrite or signal still pending")
            raw[r] = k
            flags.update(wait_s=k, wait_v=k)
            copied.append(k)
        elif op in ("wait_s", "wait_v"):
            require(flags.get(op) == k, "unpaired/wrong-generation wait")
            del flags[op]
            ready.add((op, k))
        elif op == "dequant":
            require(not failed and raw[r] == k and k == len(converted)
                    and {("wait_s", k), ("wait_v", k)} <= ready, "raw input not ready")
            require(len(accumulated) == k, "dequant before successful previous Iterate")
            require(w not in sdk_readers, "W overwrite before last SDK read drains")
            raw_readers.add(r)
            weights[w] = k
            converted.append(k)
        elif op == "v_to_s":
            require(weights[w] == k and k not in handoffs, "W is not ready for handoff")
            raw_readers.discard(r)
            handoffs.add(k)
        elif op in ("iterate", "iterate_false"):
            require(not failed and k in handoffs and weights[w] == k and k == len(accumulated)
                    and last_fenced == k - 1 and fence_due is None, "rebind/order/handoff")
            failed = op == "iterate_false"
            if not failed:
                accumulated.append(k)
                sdk_readers.add(w)
            fence_due = k
        elif op == "all":
            require(fence_due == k and (not failed or not flags), "fence order/failure drain")
            sdk_readers.clear()
            raw_readers.clear()
            last_fenced, fence_due = k, None
        else:
            raise ValueError("unknown action " + op)
    require(ended, "unfinished trace")
    return accumulated, published


def projection_calls(m, k, n):
    """Source-level work issued by all blocks for ONE projection in the accepted Raw build."""
    shape = (m, k, n)
    require(shape in {(16, 2560, 19456), (16, 9728, 2560), (16, 2560, 4096),
                      (16, 4096, 2560), (32, 2560, 2048), (80, 2560, 2048),
                      (16, 12800, 2560), (64, 12800, 2560)}, "outside the frozen projection inventory")
    stage_a = shape == (80, 2560, 2048)
    prefetch = shape in ((16, 2560, 19456), (16, 9728, 2560))
    tile_k = 128 if stage_a else 256
    n_tiles = n // 64
    steps = n_tiles * (k // tile_k)
    planes, groups = tile_k // 32, tile_k // 128
    brcb = 0 if stage_a else groups
    scalar = groups * 64 if stage_a else 0
    return Counter(steps=steps, copies_q=planes * steps, copies_s=groups * steps,
                   copies_a=int(stage_a) * steps, cast=planes * steps, brcb=brcb * steps,
                   mul=brcb * steps, get_value=scalar * steps, muls=scalar * steps,
                   explicit_pipe_v_loop=(2 + 2 * brcb) * steps,
                   explicit_pipe_v_init=0 if stage_a else 8,
                   brcb_inner_pipe_v=24 * brcb * steps, v_to_s=int(prefetch) * steps,
                   pipe_all=(1 if prefetch else 3) * steps + n_tiles,
                   iterate=steps, get_c=n_tiles)
