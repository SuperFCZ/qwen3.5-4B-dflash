"""Bind installed loader inputs before launching the newly built Gate runner."""

from pathlib import Path
import re
import subprocess

from bench import c, host, require
from scale_contracts import resume

SYSTEM = tuple(map(Path, ("/lib", "/lib64", "/usr/lib", "/usr/lib64")))


def nss_files(cache, roots=SYSTEM):
    """Resolve glibc's dlopen-only NSS files SONAME from the system loader cache."""
    result = set()
    for line in cache.splitlines():
        match = re.fullmatch(r"\s*libnss_files\.so\.2\s+\([^)]+\)\s+=>\s+(/\S+)\s*", line)
        if not match:
            continue
        alias = Path(match.group(1))
        real = alias.resolve(strict=True)
        require(
            alias.name == "libnss_files.so.2"
            and any(root in alias.parents for root in roots)
            and any(root.resolve() in real.parents for root in roots)
            and re.fullmatch(r"libnss_files(?:-[0-9]+(?:\.[0-9]+)+)?\.so(?:\.2)?", real.name),
            "NSS files binding escapes system library roots or changed basename: " + str(alias),
        )
        with real.open("rb") as handle:
            h = handle.read(20)
        require(len(h) == 20 and h[:4] == b"\x7fELF", "invalid NSS files ELF: " + str(real))
        if h[4:6] == bytes((2, 1)) and int.from_bytes(h[18:20], "little") == 183:
            result.add(real)
    return sorted(result)


def seal(folder, pkg, runner, env):
    folder.mkdir()
    cann = Path(env["CANN_ROOT"]).resolve(strict=True)
    driver = Path("/usr/local/Ascend/driver")
    opp = Path(pkg["opp"]["install"])
    rules = c.read(host.dsos.POLICY)
    extra = c.read(resume.POLICY)
    records = {}

    def bind(path, owner, kind, pin=None):
        real = path.resolve(strict=True)
        if str(real) not in records:
            records[str(real)] = host.dsos.record(real, owner.resolve(strict=True), kind)
        row = records[str(real)]
        require(row["category"] == kind, "ambiguous DSO: " + str(real))
        if pin:
            require(
                all(row["file"][k] == pin[k] for k in ("bytes", "sha256") if k in pin)
                and ("mode" not in pin or row["stat"]["mode"] == pin["mode"]),
                "installed DSO differs from pin: " + str(real),
            )
        return real

    for item in pkg["opp"]["preflight"]["files"]:
        p = Path(item["path"])
        if ".so" in p.name:
            bind(p, opp, "PRIVATE_DIAGNOSTIC_OPP", item)
    named = [(cann / "lib64" / name, cann, "STANDARD_CANN_RUNTIME", None) for name in rules["runtime_lib64"]]
    named += [
        (
            cann / "opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/aarch64" / name,
            cann,
            "STANDARD_CANN_RUNTIME",
            None,
        )
        for name in rules["builtin_host"]
    ]
    named += [(cann / i["relative"], cann, "PINNED_CANN_STANDARD_OPP", i) for i in rules["standard_opp"]]
    named += [(driver / rel, driver, "ASCEND_DRIVER", None) for rel in rules["driver"]]
    named += [
        (cann / Path(i["alias"]).relative_to(extra["cann_root"]), cann, "STANDARD_CANN_RUNTIME", i)
        for i in extra["records"]
    ]
    for p, owner, kind, pin in named:
        if p.exists():
            bind(p, owner, kind, pin)
    # NSS backends are loaded by glibc at runtime and are absent from DT_NEEDED / ldd.
    ldconfig = next((p for p in map(Path, ("/sbin/ldconfig", "/usr/sbin/ldconfig")) if p.is_file()), None)
    require(ldconfig is not None, "system ldconfig is required for NSS files binding")
    cache = subprocess.check_output([str(ldconfig), "-p"], env=env, text=True)
    nss = nss_files(cache)
    for path in nss:
        bind(path, next(root.resolve() for root in SYSTEM if root.resolve() in path.parents), "SYSTEM")
    c.write(
        folder / "nss-files.json",
        {"source": "ldconfig -p", "soname": "libnss_files.so.2", "libraries": [records[str(p)] for p in nss]},
    )
    # Bind the concrete OS dependencies of this runner and the permitted CANN/Driver DSOs.
    # No path discovered for the first time in the device process may authorize itself.
    system = set()
    deps = c.dependencies(runner, folder / "runner-ldd.log", env)
    for item in deps:
        p = Path(item["file"]["path"])
        if any(base in p.parents for base in SYSTEM):
            system.add(p)
    with (folder / "sdk-ldd.log").open("w") as log:
        for file in sorted(records):
            cp = subprocess.run(
                ["ldd", file],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            log.write(file + "\n" + cp.stdout + "\n")
            for value in re.findall(r"(?:=>\s+|^\s*)(/\S+)\s+\(", cp.stdout, re.M):
                p = Path(value).resolve(strict=True)
                if any(base in p.parents for base in SYSTEM):
                    system.add(p)
    for p in sorted(system):
        require("libcust_op" not in p.name, "foreign private op library: " + str(p))
        bind(p, next(base for base in SYSTEM if base in p.parents), "SYSTEM")
    c.write(folder / "manifest.json", records)
    return records


def check_maps(path, records):
    groups = host.dsos.maps_rows(path.read_text())
    resume.check_maps(groups, records)
    return sorted(groups)
