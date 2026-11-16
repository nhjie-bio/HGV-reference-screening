# Incremental updates on Windows

## Apply a prepared package in one click

Keep the whole prepared package together, including `payload/`, the manifest, metadata, PowerShell files and `LOCAL_REPOSITORY.txt`. Double-click **双击更新GitHub.cmd**. It finds the saved local repository, checks the files, applies the delta, commits only listed paths, pushes to `main` and verifies the remote commit. Complete GitHub sign-in if Git Credential Manager asks. A green completion message confirms the verified push.

`LOCAL_REPOSITORY.txt` contains a **local Git repository folder**, not a GitHub website URL. The package builder writes this path automatically. It is specific to your computer and is not committed to the public repository. The launcher is copied into generated update packages; it needs their manifest and payload to run.

Requirements are Git for Windows and Windows PowerShell 5.1. Applying a package needs no Python, Conda or PyTorch. Run logs are kept in the package's `logs/` directory. The optional `--check` argument performs only the read-only check. A failed check stops subsequent steps. Interrupted copies can be retried; identical installed files are skipped.

## Create the next delta

The builder requires Python 3.9+ with the standard library. Start from a clean local `main` synchronized with GitHub. Put only the intended new/modified files in a separate changes folder, preserving repository-relative paths:

```powershell
python tools/hgv_incremental/New-HGV-Delta.py --repo "D:\work\HGV-reference-screening" --changes "D:\work\W16_changes" --out "D:\work\HGV_UPDATE_W16" --id W16
```

The generated folder contains the one-click launcher and local path configuration. Add revised documentation to the curated changes folder when needed. Existing identical files and old ZIP archives are omitted. Keep raw FASTA, external pretrained weights and unrelated workspace files outside the changes folder.

## Verification and advanced use

The existing `01_CHECK.cmd`, `02_APPLY.cmd`, `03_COMMIT_AND_PUSH.cmd` and `04_VERIFY.cmd` remain available for separate operations. The one-click launcher calls the same audited updater. It refuses unexpected file contents, unrelated staged changes and divergent local/remote histories; it backs up modified files under `.git/hgv_updates/`. It supports Chinese/long Windows paths and never force-pushes or deletes repository files.

SHA256 verifies file bytes; Git blob identifiers and the remote commit verify publication. The generated `.gitattributes` preserves bytes for changed paths. If network upload fails after the update commit was created, rerunning can push that exact pending commit when the remote parent remains unchanged. Resolve any reported history conflict normally instead of forcing a push.
