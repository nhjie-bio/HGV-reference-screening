#!/usr/bin/env python3

import os
import sys
import json
import time
import signal
import argparse
import subprocess
from pathlib import Path


def pids_in_pgrp(pgid):
    out = []
    proc = Path("/proc")

    for p in proc.iterdir():
        if not p.name.isdigit():
            continue
        pid = int(p.name)
        try:
            if os.getpgid(pid) == pgid:
                out.append(pid)
        except Exception:
            pass

    return out


def pss_kib(pid):
    p = Path(f"/proc/{pid}/smaps_rollup")
    try:
        for line in p.read_text().splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1])
    except Exception:
        pass

    p = Path(f"/proc/{pid}/status")
    try:
        for line in p.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except Exception:
        pass

    return 0


def gpu_mib_for_pids(pids):
    pset = set(pids)
    try:
        x = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )

        total = 0.0

        for line in x.stdout.splitlines():
            z = [a.strip() for a in line.split(",")]
            if len(z) < 2:
                continue

            try:
                pid = int(z[0])
                mem = float(z[1])
            except Exception:
                continue

            if pid in pset:
                total += mem

        return total
    except Exception:
        return 0.0


ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sample-sec", type=float, default=0.20)
ap.add_argument("cmd", nargs=argparse.REMAINDER)
args = ap.parse_args()

cmd = list(args.cmd)
if cmd and cmd[0] == "--":
    cmd = cmd[1:]

if not cmd:
    raise SystemExit("No command supplied.")

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

stdout_path = out / "stdout.log"
stderr_path = out / "stderr.log"
json_path = out / "resource.json"

start = time.perf_counter()

with stdout_path.open("w") as fo, stderr_path.open("w") as fe:
    proc = subprocess.Popen(
        cmd,
        stdout=fo,
        stderr=fe,
        start_new_session=True,
    )

    pgid = os.getpgid(proc.pid)

    peak_pss = 0
    peak_gpu = 0.0
    samples = 0

    while True:
        pids = pids_in_pgrp(pgid)

        pss = sum(pss_kib(pid) for pid in pids)
        gpu = gpu_mib_for_pids(pids)

        peak_pss = max(peak_pss, pss)
        peak_gpu = max(peak_gpu, gpu)
        samples += 1

        rc = proc.poll()

        if rc is not None:
            # one last sample
            pids = pids_in_pgrp(pgid)
            pss = sum(pss_kib(pid) for pid in pids)
            gpu = gpu_mib_for_pids(pids)

            peak_pss = max(peak_pss, pss)
            peak_gpu = max(peak_gpu, gpu)
            break

        time.sleep(args.sample_sec)

end = time.perf_counter()

data = {
    "name": args.name,
    "command": cmd,
    "exit_code": int(proc.returncode),
    "wall_s": float(end - start),
    "peak_host_pss_kib": int(peak_pss),
    "peak_host_pss_mib": float(peak_pss / 1024.0),
    "peak_gpu_mib": float(peak_gpu),
    "samples": int(samples),
    "sample_interval_s": float(args.sample_sec),
    "memory_definition":
        "sum of /proc/<pid>/smaps_rollup Pss within spawned process group",
}

json_path.write_text(json.dumps(data, indent=2) + "\n")

print(
    f"{args.name}\t"
    f"rc={proc.returncode}\t"
    f"wall={data['wall_s']:.6f}s\t"
    f"host_pss={data['peak_host_pss_mib']:.3f}MiB\t"
    f"gpu={data['peak_gpu_mib']:.3f}MiB"
)

if proc.returncode != 0:
    print("STDERR:")
    try:
        print(stderr_path.read_text()[-10000:])
    except Exception:
        pass

    raise SystemExit(proc.returncode)
