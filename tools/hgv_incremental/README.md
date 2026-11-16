# Incremental updates on Windows

The updater requires Windows PowerShell 5.1 or PowerShell 7 and Git for Windows. Applying a prepared update needs no Python, PyTorch or training environment. Only the future-package builder requires Python 3.9+ (standard library).

1. Keep a normal clone of this repository on `main`, synchronized with GitHub.
2. For a future experiment, place only the intended new/modified files in a separate changes folder, preserving repository-relative paths. Do not point the builder at the entire original research project.
3. Build the delta from a clean synchronized repository:

```powershell
python tools/hgv_incremental/New-HGV-Delta.py --repo "D:\work\HGV-reference-screening" --changes "D:\work\W16_changes" --out "D:\work\HGV_UPDATE_W16" --id W16
```

For example, `W16_changes/HGV_V3_revision_20260915/scripts_frontiers_windows/W16_example.py` and the corresponding result files become additions under those exact paths. Supply a revised documentation file in the changes folder if it should be updated; unrelated files and old ZIP archives are excluded.

4. Run `01_CHECK.cmd "D:\work\HGV-reference-screening"`, then `02_APPLY.cmd` with the same argument. Review the working tree and `UPDATE_MANIFEST.tsv`.
5. Run `03_COMMIT_AND_PUSH.cmd` with the repository path when ready to publish. This explicitly commits and pushes to `main`; it is separate from local application. Existing Git credentials are used. The updater never requests or prints a token.
6. Run `04_VERIFY.cmd` to check installed SHA256 values. Publish additionally verifies every committed blob and the remote `main` commit.

The updater accepts add/modify actions only. It rejects path traversal, reparse points, unexpected existing-file hashes, unrelated staged changes and divergent local/remote histories. It backs up modified files under `.git/hgv_updates/`, checks bytes after copying, stages only manifest paths (including explicitly listed archived logs ignored by `.gitignore`) and never force-pushes. A partial apply can be rerun; files with the expected new hash are skipped. After a push failure the same update can retry its own pending commit if the remote parent is unchanged.

`Plan` is read-only. `Apply` changes the local working tree only. `Publish` requires the full update to be installed. Untracked files outside the manifest are not staged. If conflicts are detected, resolve them deliberately and rebuild the delta against the newer base. There is no automatic reset, deletion or rebase.

SHA256 identifies installed bytes; Git blob identifiers identify committed content. `.gitattributes` preserves exact bytes across Windows/Linux checkouts. Archive-level historical checksum files remain provenance records; the update manifest is the verification source for a prepared incremental package.
