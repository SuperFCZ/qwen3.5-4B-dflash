"""Reuse the admitted CANN CPU model and bind its layout witness to each layer's own scale."""

from evidence import FIRST, LAYERS, c, require
from scale_contracts import command, host, sdk_files
from scale_review import expected_line


def review_scales(folder, ctx, chain, gear, *, execute):
    src = ctx["src"]
    sdk = ctx["pilot"]["review"]["sdk_cpu"]
    require(sdk["sdk"] == sdk_files(src["cann"]), "CANN CPU model changed")
    binary = c.verify(sdk["binary"])
    prior = c.read(c.verify(src["host_source"]))["binding"]["prior"]
    env, _ = host.envs.for_prior(prior, None)
    require(not env["ASCEND_CUSTOM_OPP_PATH"], "unexpected OPP in CPU model")
    if execute:
        require(
            c.dependencies(sdk["binary"], folder / "ldd.log", env) == sdk["dependencies"],
            "scale CPU DSO drift",
        )
    witnesses = {}
    for layer in LAYERS:
        item = src["cases"][gear][layer]["inputs"]["s_gn.bin"]
        scale = c.verify(item)
        require(item["bytes"] == 389120 * 2, "wrong Group128 scale extent")
        if gear == "64":
            ref = chain["c16-review"]["scale_words"][layer]
            require(
                (item["bytes"], item["sha256"]) == (ref["input"]["bytes"], ref["input"]["sha256"]),
                "C64 has another scale payload for " + layer,
            )
            run, origin = ref["run"], "SAME_LAYER_IDENTICAL_SCALE_WORDS"
            executed_input = ref["executed_input"]
        elif layer == FIRST:
            run, executed_input, origin = sdk["frozen"], sdk["frozen_scale"], "FROZEN_PILOT"
        else:
            dest = folder / layer
            log = dest / "model.log"
            if execute:
                dest.mkdir()
                command([binary, scale], log, dict(env, TMPDIR=str(dest), LC_ALL="C"), cwd=dest)
            run = dict(log=c.identity(log), command=c.identity(log.with_suffix(".command.json")))
            executed_input, origin = item, "CHECKED_THIS_LAYER"
        require(
            executed_input["sha256"] == item["sha256"]
            and c.read(c.verify(run["command"]))["argv"] == [str(binary), str(c.verify(executed_input))]
            and c.verify(run["log"]).read_text(errors="replace").splitlines().count(expected_line(389120))
            == 1,
            "missing verified SDK layout witness: " + layer,
        )
        witnesses[layer] = dict(
            input=item,
            executed_input=executed_input,
            run=run,
            source_words=389120,
            expanded_words=6225920,
            origin=origin,
            cpu_binary=sdk["binary"],
        )
    return witnesses
