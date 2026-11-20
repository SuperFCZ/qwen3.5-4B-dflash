"""CPU partial-order model for the frozen M16/K2560/N19456 Gate. No device API.

Nodes distinguish starting an operation from retiring it. Two commands merely
issued to the same pipe do not get a completion edge. A barrier or a directed
event supplies one. Each modeled transfer groups all slices of one operand.
"""

from dataclasses import dataclass

PIPES = ("S", "V", "M", "MTE1", "MTE2", "MTE3")
ALL = "PIPE_ALL"
MTE3 = "PIPE_MTE3"
W_EVENT = "MTE3_V"
NARROW = "double-w-overlap-narrow-fence"
DESIGN = ("single-w-raw", "double-w-serial", "double-w-overlap", NARROW)


def reserve_fence_ids(occupied, sdk_spares):
    """Model dav_2002 per-direction sff0/Alloc with all-or-nothing ownership.

    The caller supplies the SDK demand from the active route, not a guessed
    physical ID. On failure both application reservations must be rolled back.
    """
    new = dict(occupied)
    leases = {}
    for direction in ("MTE1_V", "V_MTE2"):
        mask, needed = new[direction], sdk_spares[direction]
        if type(mask) is not int or not 0 <= mask < 256 or type(needed) is not int or needed < 0:
            raise ValueError("invalid event pool or SDK demand")
        free = [slot for slot in range(8) if not mask & (1 << slot)]
        if len(free) <= needed:
            return dict(occupied), {}
        slot = free[0]
        new[direction] |= 1 << slot
        leases[direction] = slot
    return new, leases


class Queues:
    """Vector clocks represented by bit sets of explicitly completed nodes."""

    def __init__(self):
        # Completion nodes of commands already issued; they might still be in flight.
        self.issued_ends = {p: 0 for p in PIPES}
        self.gate = {p: 0 for p in PIPES}
        self.clock = {}
        self.bit = {}
        self.signals = {}
        self.waits = {}
        self.rearms = []

    def node(self, name, before):
        if name in self.clock:
            raise ValueError("duplicate operation: " + name)
        bit = 1 << len(self.bit)
        self.bit[name] = bit
        self.clock[name] = before | bit
        return self.clock[name]

    def op(self, name, pipe):
        # The scalar issuer must pass any earlier explicit wait on S to submit this operation.
        start = self.node(name + ".start", self.gate[pipe] | self.gate["S"])
        self.issued_ends[pipe] |= self.node(name + ".end", start)

    def before(self, earlier, later):
        return earlier != later and bool(self.bit[earlier] & self.clock[later])

    def barrier(self, name, pipe):
        pipes = PIPES if pipe == "ALL" else (pipe,)
        before = self.gate["S"]
        for p in pipes:
            before |= self.issued_ends[p] | self.gate[p]
        point = self.node(name, before)
        for p in pipes:
            self.gate[p] = point

    def signal(self, name, src, dst, slot):
        # "slot" is a logical reservation, never a hardware ID or an assumed ID value.
        key = (src, dst, slot)
        point = self.node(name + ".set", self.issued_ends[src] | self.gate[src] | self.gate["S"])
        if key in self.waits:
            earlier = self.waits[key]
            self.rearms.append((key, earlier, name + ".set", self.before(earlier, name + ".set")))
        self.issued_ends[src] |= point
        self.signals[name] = (key, point)

    def wait(self, name):
        key, point = self.signals.pop(name)
        dst = key[1]
        point = self.node(name + ".wait", point | self.gate[dst] | self.gate["S"])
        self.gate[dst] = point
        self.issued_ends[dst] |= point
        self.waits[key] = name + ".wait"

    def pair(self, name, src, dst, slot):
        self.signal(name, src, dst, slot)
        self.wait(name)


@dataclass
class Trace:
    queues: Queues
    actions: list
    cubes: list
    outputs: list
    drains: list


def simulate(
    layout="double-w-overlap",
    fence=ALL,
    *,
    n_tiles=1,
    fail=None,
    failure_site="MoveNext",
    matrix_barrier=True,
):
    """'fail' is (within-core N ordinal, K). No timing or hardware simulation.

    remove-one-edge variants are test mutations only. All designs retain the
    success tail, N and ordinary failure full barriers. SDK queue witnesses are
    aggregated on the active, aligned normal path, including the ND A queue.
    """
    options = (ALL, MTE3, W_EVENT, NARROW, "MTE1_V-only", "V_MTE2-only")
    if layout not in DESIGN or fence not in options or n_tiles <= 0:
        raise ValueError("unsupported design")
    if layout in DESIGN[:2] and fence != ALL:
        raise ValueError("only overlap steady-state fences are under review")
    if layout == NARROW and fence != NARROW:
        raise ValueError("wrong candidate policy")
    if fail is not None and not (0 <= fail[0] < n_tiles and 0 <= fail[1] < 10):
        raise ValueError("invalid failure position")
    if failure_site not in ("MoveNext", "V_S-unavailable"):
        raise ValueError("unknown failure")
    if layout == "single-w-raw" and fail is not None and failure_site == "V_S-unavailable":
        raise ValueError("the frozen single-W source has no checked V_S failure branch")
    g = Queues()
    trace = Trace(g, [], [], [], [])
    g.op("init.expanded", "V")
    g.barrier("init.PIPE_V", "V")
    last_slot = {}  # primed SDK M->MTE1 credits for ping and pong
    for slot in (0, 1):
        last_slot[slot] = f"init.L0.{slot}"
        g.signal(last_slot[slot], "M", "MTE1", slot)

    def end():
        g.op("release_end", "S")  # application release, mm.End; not an extra completion fence
        for token in last_slot.values():
            g.wait(token)  # SDK destructor consumes its own credits

    for n in range(n_tiles):

        def tag(k, what):
            return f"n{n}.k{k}.{what}"

        def act(k, what):
            trace.actions.append((n, k, what))

        def copy(k):
            if not 0 <= k < 10:
                raise ValueError("raw DMA out of bounds")
            act(k, "copy")
            g.op(tag(k, "raw"), "MTE2")
            for dst in ("S", "V"):
                g.signal(tag(k, "raw_" + dst), "MTE2", dst, "raw")

        def wait_raw(k):
            act(k, "wait_raw")
            for dst in ("S", "V"):
                g.wait(tag(k, "raw_" + dst))

        def dequant(k):
            act(k, "dequant")
            # This op retires the whole D, with six outer and 48 Brcb internal PIPE_Vs.
            g.op(tag(k, "D"), "V")
            g.barrier(tag(k, "last_PIPE_V"), "V")

        copy(0)
        if layout != "single-w-raw":
            wait_raw(0)
            dequant(0)
        for k in range(10):
            next_k = k + 1
            if layout == "single-w-raw":
                wait_raw(k)
            if next_k < 10:
                copy(next_k)
            if layout == "single-w-raw":
                dequant(k)
            failure = fail == (n, k)
            if not (failure and failure_site == "V_S-unavailable"):
                g.pair(tag(k, "ready"), "V", "S", "ready")
                g.op(tag(k, "bind"), "S")
                act(k, "iterate")
            if failure:
                if next_k < 10:
                    wait_raw(next_k)
                g.barrier(tag(k, "failure_ALL"), "ALL")
                trace.drains.append(tag(k, "failure_ALL"))
                end()
                return trace
            # k resets both cache positions. Depth one recycles the A1/B1 storage.
            # Both reverse queues target MTE3 due to the ND flag on dav_2002.
            if n or k:
                for operand in ("A", "B"):
                    g.signal(tag(k, "free_" + operand), "MTE1", "MTE3", "queue_" + operand)
                g.wait(tag(k, "free_A"))
            g.op(tag(k, "A1_write"), "MTE2")
            g.pair(tag(k, "A_copy_ready"), "MTE2", "MTE1", "A_copy")
            g.pair(tag(k, "A_queue_ready"), "MTE3", "MTE1", "A_queue")
            if n or k:
                g.wait(tag(k, "free_B"))
            g.op(tag(k, "B1_write_W_read"), "MTE3")
            g.pair(tag(k, "B_queue_ready"), "MTE3", "MTE1", "B_queue")
            slot = (10 * n + k) % 2
            g.wait(last_slot[slot])
            for operand in ("A", "B"):
                g.op(tag(k, operand + "1_load_L0"), "MTE1")
            g.pair(tag(k, "L0_ready"), "MTE1", "M", slot)
            g.op(tag(k, "mad"), "M")
            trace.cubes.append((n, k, k != 0))
            if matrix_barrier:
                g.barrier(tag(k, "SDK_PIPE_M"), "M")
            last_slot[slot] = tag(k, "L0_free")
            g.signal(last_slot[slot], "M", "MTE1", slot)
            if layout != "single-w-raw" and next_k < 10:
                wait_raw(next_k)
            if layout in ("double-w-overlap", NARROW) and next_k < 10:
                dequant(next_k)
            if k == 9 or fence == ALL:
                g.barrier(tag(k, "F_ALL"), "ALL")
            elif fence == MTE3:
                g.barrier(tag(k, "F_MTE3"), "MTE3")
            elif fence == W_EVENT:
                g.pair(tag(k, "F_W"), "MTE3", "V", "candidate_W")
            else:
                if fence in (NARROW, "MTE1_V-only"):
                    g.pair(tag(k, "F_load"), "MTE1", "V", "candidate_load")
                if fence in (NARROW, "V_MTE2-only"):
                    g.pair(tag(k, "F_raw"), "V", "MTE2", "candidate_raw")
            act(k, "fence")
            if layout == "double-w-serial" and next_k < 10:
                dequant(next_k)
        # Final copy is CO1 -> UB -> GM; its public SDK handoffs are retained.
        g.pair(tag(9, "CO1_ready"), "M", "V", "CO1")
        g.pair(tag(9, "copyout_enter"), "MTE3", "V", "SDK_copyout")
        g.op(tag(9, "fp16_CO2"), "V")
        g.pair(tag(9, "copyout_ready"), "V", "MTE3", "SDK_copyout")
        g.op(tag(9, "GM_out"), "MTE3")
        g.pair(tag(9, "copyout_done"), "MTE3", "V", "SDK_copyout")
        g.barrier(tag(9, "N_ALL"), "ALL")
        trace.outputs.append(n)
        trace.drains.append(tag(9, "N_ALL"))
    end()
    return trace


def missing_edges(trace, banks):
    """List absent causal edges, including WAR on A1 and the independent L0C RAW edge."""
    g = trace.queues
    missing = []
    for n, k, _ in trace.cubes:

        def check(name, left, right):
            if left in g.bit and right in g.bit and not g.before(left, right):
                missing.append((name, left, right))

        prefix = f"n{n}.k"
        cur = prefix + str(k) + "."
        check("raw ready", cur + "raw.end", cur + "D.start")
        check("W ready", cur + "D.end", cur + "B1_write_W_read.start")
        check("W reuse", cur + "B1_write_W_read.end", prefix + str(k + banks) + ".D.start")
        check("raw reuse", cur + "D.end", prefix + str(k + 2) + ".raw.start")
        for op, write in (("A", "A1_write"), ("B", "B1_write_W_read")):
            check(op + "1 reuse", cur + op + "1_load_L0.end", prefix + str(k + 1) + "." + write + ".start")
            check("L0" + op + " reuse", cur + "mad.end", prefix + str(k + 2) + "." + op + "1_load_L0.start")
        check("L0C partial", cur + "mad.end", prefix + str(k + 1) + ".mad.start")
        check("final copy", cur + "mad.end", prefix + "9.fp16_CO2.start")
        check("cross N output", prefix + "9.GM_out.end", f"n{n + 1}.k0.raw.start")
        if k == 9:
            for left, right in (
                ("mad", "mad"),
                ("fp16_CO2", "mad"),
                ("A1_load_L0", "A1_write"),
                ("B1_load_L0", "B1_write_W_read"),
                ("B1_write_W_read", "D"),
            ):
                check("cross N ownership", cur + left + ".end", f"n{n + 1}.k0.{right}.start")
    return missing
