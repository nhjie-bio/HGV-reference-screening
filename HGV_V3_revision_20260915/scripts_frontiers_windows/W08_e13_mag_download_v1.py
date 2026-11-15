#!/usr/bin/env python3
import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

SHORTLIST = BASE / (
    "21_frontiers_revision_20261112/"
    "90_windows_minimal/W06_E13_MAG/"
    "W06_E13_MAG_SHORTLIST.tsv"
)

MASTER = BASE / (
    "01_manifest/final_3154_freeze_v1_20260916/"
    "final_master_manifest_3154.tsv"
)

OUT = BASE / (
    "21_frontiers_revision_20261112/"
    "90_windows_minimal/W08_E13_MAG_FASTA"
)

BINS = ("50-70", "70-80", "80-90", "90-100")

URL_ROOT = (
    "https://api.ncbi.nlm.nih.gov/datasets/v2/"
    "genome/accession/"
)

MAX_ZIP = 100 * 1024 * 1024
MAX_FASTA = 120 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise RuntimeError("STOP: " + message)


def load(path):
    require(path.is_file(), "Missing " + str(path))

    with path.open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(
            handle, delimiter="\t"
        )
        rows = list(reader)

    require(
        not any(None in r for r in rows),
        "Malformed TSV " + str(path)
    )

    return rows


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1 << 20), b""
        ):
            h.update(block)

    return h.hexdigest()


def write_tsv(path, rows, fields):
    tmp = path.with_suffix(
        path.suffix + ".part"
    )

    with tmp.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        w = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n"
        )
        w.writeheader()
        w.writerows(rows)

    os.replace(tmp, path)


def validate_candidates():

    rows = load(SHORTLIST)

    require(
        len(rows) == 80,
        "W06 shortlist is not 80 rows"
    )

    accs = [
        r["accession"].strip()
        for r in rows
    ]

    require(
        len(set(accs)) == 80,
        "Duplicate MAG accession"
    )

    biosamples = [
        r["biosample"].strip()
        for r in rows
        if r["biosample"].strip()
    ]

    require(
        len(set(biosamples)) == len(biosamples),
        "Duplicate MAG BioSample"
    )

    original = load(MASTER)

    used_acc = {
        r["assembly_accession_final"].strip()
        for r in original
    }

    used_bio = {
        r["biosample"].strip()
        for r in original
        if r["biosample"].strip()
    }

    for r in rows:

        acc = r["accession"].strip()

        require(
            re.fullmatch(
                r"GC[AF]_\d+\.\d+", acc
            ) is not None,
            "Invalid accession " + acc
        )

        require(
            acc not in used_acc,
            "Accession overlaps original: " + acc
        )

        require(
            not r["biosample"].strip()
            or r["biosample"].strip() not in used_bio,
            "BioSample overlaps original: " + acc
        )

        require(
            r["ncbi_genome_category"].strip().lower()
            == "derived from metagenome",
            "Not labeled MAG: " + acc
        )

        contamination = float(
            r["contamination"]
        )

        completeness = float(
            r["completeness"]
        )

        require(
            0 <= contamination <= 5,
            "Contamination out of range: " + acc
        )

        b = r["quality_bin"]

        valid_bin = (
            (b == "50-70" and 50 <= completeness < 70)
            or
            (b == "70-80" and 70 <= completeness < 80)
            or
            (b == "80-90" and 80 <= completeness < 90)
            or
            (b == "90-100" and 90 <= completeness <= 100)
        )

        require(
            valid_bin,
            "Completeness bin mismatch: " + acc
        )

    for b in BINS:

        group = [
            r for r in rows
            if r["quality_bin"] == b
        ]

        require(
            len(group) == 20,
            "Wrong bin size: " + b
        )

        levels = Counter(
            r["reference_target_level"]
            for r in group
        )

        require(
            levels == Counter({
                "species": 10,
                "genus": 6,
                "family": 4
            }),
            "Wrong taxonomic quota: " + b
        )

    return rows


def choose_pilot(rows):

    selected = []

    for b in BINS:

        group = [
            r for r in rows
            if r["quality_bin"] == b
        ]

        for level in ("species", "genus"):

            selected.append(
                next(
                    r for r in group
                    if r["reference_target_level"] == level
                )
            )

    return selected


def download_zip(acc, destination):

    if destination.exists():
        return "REUSED"

    url = (
        URL_ROOT
        + urllib.parse.quote(acc)
        + "/download?include_annotation_type=GENOME_FASTA"
    )

    tmp = destination.with_suffix(
        ".zip.part"
    )

    for attempt in range(1, 4):

        try:

            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/zip",
                    "User-Agent":
                        "HGV-V3-academic-revision/1.0"
                }
            )

            with urllib.request.urlopen(
                req, timeout=45
            ) as reply, tmp.open("wb") as handle:

                total = 0

                while True:

                    chunk = reply.read(1 << 20)

                    if not chunk:
                        break

                    total += len(chunk)

                    require(
                        total <= MAX_ZIP,
                        "Downloaded package exceeds 100 MiB"
                    )

                    handle.write(chunk)

            require(
                zipfile.is_zipfile(tmp),
                "Response is not ZIP: " + acc
            )

            os.replace(tmp, destination)

            return "DOWNLOADED"

        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            OSError,
            RuntimeError
        ) as exc:

            if tmp.exists():
                tmp.unlink()

            if (
                isinstance(exc, urllib.error.HTTPError)
                and exc.code not in
                (408, 429, 500, 502, 503, 504)
            ):
                raise RuntimeError(
                    "HTTP " + str(exc.code)
                    + " for " + acc
                ) from exc

            if attempt == 3:
                raise RuntimeError(
                    "Download failed: " + repr(exc)
                ) from exc

            time.sleep(2 * attempt)

    raise RuntimeError(
        "Unexpected download state"
    )


def extract_exact(acc, archive, destination):

    with zipfile.ZipFile(archive, "r") as z:

        require(
            z.testzip() is None,
            "ZIP CRC failure: " + acc
        )

        fasta_members = [
            item
            for item in z.infolist()
            if not item.is_dir()
            and item.filename.lower().endswith(".fna")
            and (
                "/" + acc + "/"
            ) in (
                "/" + item.filename.replace("\\", "/")
            )
        ]

        require(
            len(fasta_members) == 1,
            "Expected exactly one FASTA for "
            + acc + ", found "
            + str(len(fasta_members))
        )

        info = fasta_members[0]

        require(
            info.file_size <= MAX_FASTA,
            "Uncompressed FASTA exceeds 120 MiB"
        )

        reports = [
            item
            for item in z.infolist()
            if item.filename.endswith(
                "assembly_data_report.jsonl"
            )
        ]

        require(
            len(reports) == 1,
            "Assembly report missing or ambiguous"
        )

        with z.open(reports[0]) as handle:

            records = [
                json.loads(line)
                for line in handle
                if line.strip()
            ]

        require(
            any(
                r.get("accession") == acc
                for r in records
            ),
            "NCBI assembly report accession mismatch: "
            + acc
        )

        h = hashlib.sha256()
        total = 0

        if not destination.exists():

            tmp = destination.with_suffix(
                ".fna.gz.part"
            )

            try:

                first = True

                with z.open(info) as src, \
                     tmp.open("wb") as raw_output:

                    with gzip.GzipFile(
                        fileobj=raw_output,
                        mode="wb",
                        filename="",
                        mtime=0
                    ) as dst:

                        for chunk in iter(
                            lambda: src.read(1 << 20),
                            b""
                        ):

                            if first:
                                require(
                                    chunk.startswith(b">"),
                                    "Invalid FASTA: " + acc
                                )
                                first = False

                            h.update(chunk)
                            total += len(chunk)
                            dst.write(chunk)

                require(
                    total == info.file_size,
                    "FASTA size mismatch"
                )

                os.replace(tmp, destination)

            finally:
                if tmp.exists():
                    tmp.unlink()

        else:

            with z.open(info) as src:

                for chunk in iter(
                    lambda: src.read(1 << 20), b""
                ):
                    h.update(chunk)
                    total += len(chunk)

            require(
                total == info.file_size,
                "ZIP FASTA size mismatch"
            )

            existing = hashlib.sha256()

            with gzip.open(
                destination, "rb"
            ) as handle:

                for chunk in iter(
                    lambda: handle.read(1 << 20),
                    b""
                ):
                    existing.update(chunk)

            require(
                existing.hexdigest() == h.hexdigest(),
                "Existing FASTA differs from ZIP: "
                + acc
            )

        require(
            destination.is_file(),
            "FASTA missing after extraction"
        )

        return h.hexdigest(), total


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--mode",
        choices=("pilot", "all"),
        required=True
    )

    args = ap.parse_args()

    rows = validate_candidates()

    selection = (
        choose_pilot(rows)
        if args.mode == "pilot"
        else rows
    )

    OUT.mkdir(
        parents=True, exist_ok=True
    )

    archives = OUT / "ncbi_packages"
    fastas = OUT / "genomic_fasta_gz"

    archives.mkdir(exist_ok=True)
    fastas.mkdir(exist_ok=True)

    fields = list(rows[0]) + [
        "download_status",
        "fasta_gz_path",
        "ncbi_zip_path",
        "ncbi_zip_sha256",
        "fna_sha256",
        "fna_bytes",
        "error"
    ]

    results = []
    consecutive_failures = 0

    print(
        "W08 E13 REAL MAG FASTA DOWNLOAD",
        "mode=" + args.mode,
        "targets=" + str(len(selection)),
        flush=True
    )

    for idx, r in enumerate(selection, 1):

        acc = r["accession"]

        archive = archives / (acc + ".zip")
        fasta = fastas / (acc + ".fna.gz")

        output = dict(r)

        output.update({
            "download_status": "FAIL",
            "fasta_gz_path": "",
            "ncbi_zip_path": "",
            "ncbi_zip_sha256": "",
            "fna_sha256": "",
            "fna_bytes": "",
            "error": ""
        })

        try:

            mode = download_zip(
                acc, archive
            )

            digest, size = extract_exact(
                acc, archive, fasta
            )

            output.update({
                "download_status": "PASS_" + mode,
                "fasta_gz_path": str(fasta),
                "ncbi_zip_path": str(archive),
                "ncbi_zip_sha256": sha256(archive),
                "fna_sha256": digest,
                "fna_bytes": size
            })

            consecutive_failures = 0

            print(
                f"[{idx}/{len(selection)}] PASS "
                f"{acc} {r['quality_bin']} "
                f"{r['reference_target_level']} "
                f"bytes={size}",
                flush=True
            )

        except Exception as exc:

            output["error"] = str(exc)

            consecutive_failures += 1

            print(
                f"[{idx}/{len(selection)}] FAIL "
                f"{acc}: {exc}",
                flush=True
            )

        results.append(output)

        write_tsv(
            OUT / (
                "W08_" + args.mode.upper()
                + "_AUDIT.tsv"
            ),
            results,
            fields
        )

        if consecutive_failures >= 3:

            print(
                "[STOP] Three consecutive failures; "
                "stopping further requests.",
                flush=True
            )

            break

        time.sleep(0.5)

    passed = [
        r for r in results
        if r["download_status"].startswith("PASS_")
    ]

    write_tsv(
        OUT / (
            "W08_" + args.mode.upper()
            + "_VALIDATED_MAG.tsv"
        ),
        passed,
        fields
    )

    success = len(passed) == len(selection)

    status = (
        "MODE=" + args.mode
        + "\nTARGET=" + str(len(selection))
        + "\nPASS=" + str(len(passed))
        + "\nFAILED=" + str(len(results) - len(passed))
        + "\nSTATUS="
        + ("PASS" if success else "INCOMPLETE")
        + "\nNO_MODEL_TRAINING=TRUE\n"
    )

    (
        OUT / (
            "W08_" + args.mode.upper()
            + "_STATUS.txt"
        )
    ).write_text(
        status, encoding="utf-8"
    )

    print()
    print(
        "W08_" + args.mode.upper()
        + "_STATUS="
        + ("PASS" if success else "INCOMPLETE")
    )

    print(
        "Downloaded and validated:",
        len(passed), "/", len(selection)
    )

    print("Output:", OUT)

    if not success:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
