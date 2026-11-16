[CmdletBinding()]
param(
    [string]$RepoPath,
    [ValidateSet('Plan','Apply','Verify','Publish')][string]$Mode = 'Plan'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$utf8 = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = $utf8
[Console]::OutputEncoding = $utf8

function Native-Path([string]$Path) {
    if ($Path.StartsWith('\\?\')) { return $Path }
    if ($Path.StartsWith('\\')) { return '\\?\UNC\' + $Path.Substring(2) }
    return '\\?\' + $Path
}
function Hash([string]$Path) {
    return (Get-FileHash -LiteralPath (Native-Path $Path) -Algorithm SHA256).Hash.ToLowerInvariant()
}
function Old-Matches([string]$Path, $Row) {
    if ((Hash $Path) -eq $Row.old_sha256) { return $true }
    # A normal Windows checkout may convert LF to CRLF. Accept that exact
    # text-only difference, but never accept edits to the actual content.
    if ($Row.PSObject.Properties.Name -contains 'old_lf_sha256' -and $Row.old_lf_sha256) {
        try {
            $strict = New-Object System.Text.UTF8Encoding($false,$true)
            $text = $strict.GetString([IO.File]::ReadAllBytes($Path)).Replace("`r`n","`n")
            $hash = [Security.Cryptography.SHA256]::Create()
            try { $value = ([BitConverter]::ToString($hash.ComputeHash($strict.GetBytes($text)))).Replace('-','').ToLowerInvariant() }
            finally { $hash.Dispose() }
            return ($value -eq $Row.old_lf_sha256)
        } catch { return $false }
    }
    return $false
}
function Git([string[]]$Arguments) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & git.exe -c core.longpaths=true -C $script:repo @Arguments 2>&1
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $old }
    $result = [string]::Join("`n", [string[]]@($output))
    if ($code -ne 0) { throw "git $($Arguments[0]) failed ($code): $result" }
    return $result
}
function Normalize-Remote([string]$Value) {
    return (($Value.Trim() -replace '^git@github.com:', 'https://github.com/').TrimEnd('/') -replace '\.git$', '')
}
function Safe-Path([string]$Root, [string]$Relative) {
    if (!$Relative -or $Relative.Contains('\') -or $Relative -match '(^/|:|(^|/)\.\.?(/|$)|(^|/)\.git(/|$)|[\x00-\x1f])') {
        throw "Unsafe manifest path: $Relative"
    }
    $full = [IO.Path]::GetFullPath([IO.Path]::Combine($Root, $Relative.Replace('/', [IO.Path]::DirectorySeparatorChar)))
    if (!$full.StartsWith($Root.TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes root: $Relative"
    }
    $walk = $full
    while ($walk.Length -ge $Root.Length) {
        if (Test-Path -LiteralPath (Native-Path $walk)) {
            $item = Get-Item -LiteralPath (Native-Path $walk) -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point refused: $walk" }
        }
        if ($walk -eq $Root) { break }
        $walk = [IO.Path]::GetDirectoryName($walk)
    }
    return (Native-Path $full)
}
function Verify-Payload {
    $seen = @{}
    foreach ($row in $script:rows) {
        if ($seen.ContainsKey($row.path)) { throw "Duplicate manifest path: $($row.path)" }
        $seen[$row.path] = $true
        if ($row.action -notin @('ADD','MODIFY')) { throw "Unsupported action: $($row.action)" }
        if ($row.sha256 -notmatch '^[0-9a-f]{64}$') { throw "Bad SHA256: $($row.path)" }
        $source = Safe-Path $script:payload $row.path
        if (!(Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing payload: $($row.path)" }
        if ((Get-Item -LiteralPath $source).Length -ne [long]$row.bytes -or (Hash $source) -ne $row.sha256) {
            throw "Payload checksum mismatch: $($row.path)"
        }
        if ([long]$row.bytes -ge 100MB) { throw "File exceeds ordinary GitHub file limit: $($row.path)" }
    }
}
function Get-Plan {
    $result = @()
    foreach ($row in $script:rows) {
        $dest = Safe-Path $script:repo $row.path
        $state = 'ADD'
        if (Test-Path -LiteralPath $dest) {
            if (!(Test-Path -LiteralPath $dest -PathType Leaf)) { throw "File/directory conflict: $($row.path)" }
            $current = Hash $dest
            if ($current -eq $row.sha256) { $state = 'UNCHANGED' }
            elseif ($row.action -eq 'MODIFY' -and (Old-Matches $dest $row)) { $state = 'MODIFY' }
            else { throw "Existing file differs; nothing overwritten: $($row.path)" }
        } elseif ($row.action -eq 'MODIFY') { throw "Expected old file missing: $($row.path)" }
        $result += [pscustomobject]@{path=$row.path; state=$state; destination=$dest; row=$row}
    }
    return $result
}
function Verify-Installed {
    foreach ($row in $script:rows) {
        $dest = Safe-Path $script:repo $row.path
        if (!(Test-Path -LiteralPath $dest -PathType Leaf) -or (Hash $dest) -ne $row.sha256) {
            throw "Installed checksum mismatch: $($row.path)"
        }
    }
}
function New-ReportDir {
    $gitDir = Git -Arguments @('rev-parse','--absolute-git-dir')
    $dir = Join-Path $gitDir ('hgv_updates/' + $script:meta.update_id + '/' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
    [IO.Directory]::CreateDirectory($dir) | Out-Null
    return $dir
}
function Verify-Commit {
    # Compare the exact bytes with committed Git blobs, independent of line-ending settings.
    $tree = @{}
    foreach ($line in (Git -Arguments @('-c','core.quotepath=false','ls-tree','-r','HEAD')).Split("`n")) {
        if ($line -match '^\d+\s+blob\s+([0-9a-f]+)\t(.*)$') { $tree[$Matches[2]] = $Matches[1] }
    }
    $format = Git -Arguments @('rev-parse','--show-object-format')
    if ($format -ne 'sha1') { throw 'This updater currently verifies SHA-1 Git repositories only (payload checks use SHA256).' }
    foreach ($row in $script:rows) {
        $file = Safe-Path $script:repo $row.path
        $hasher = [Security.Cryptography.SHA1]::Create()
        $stream = [IO.File]::OpenRead($file)
        try {
            $header = [Text.Encoding]::ASCII.GetBytes(('blob ' + $stream.Length + [char]0))
            $hasher.TransformBlock($header,0,$header.Length,$null,0) | Out-Null
            $buffer = New-Object byte[] 65536
            while (($count = $stream.Read($buffer,0,$buffer.Length)) -gt 0) {
                $hasher.TransformBlock($buffer,0,$count,$null,0) | Out-Null
            }
            $hasher.TransformFinalBlock((New-Object byte[] 0),0,0) | Out-Null
            $actual = ([BitConverter]::ToString($hasher.Hash)).Replace('-','').ToLowerInvariant()
        } finally { $stream.Dispose(); $hasher.Dispose() }
        if (!$tree.ContainsKey($row.path) -or $actual -ne $tree[$row.path]) { throw "Committed blob mismatch: $($row.path)" }
    }
}

try {
    $metaPath = Join-Path $PSScriptRoot 'UPDATE_METADATA.json'
    $script:meta = Get-Content -LiteralPath $metaPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $manifest = Join-Path $PSScriptRoot 'UPDATE_MANIFEST.tsv'
    if ((Hash $manifest) -ne $meta.manifest_sha256) { throw 'UPDATE_MANIFEST.tsv checksum mismatch.' }
    $script:rows = @(Import-Csv -LiteralPath $manifest -Delimiter "`t" -Encoding UTF8)
    if ($rows.Count -ne $meta.file_count) { throw 'Manifest count mismatch.' }
    $script:payload = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'payload')).TrimEnd('\','/')
    if (!$RepoPath) {
        $savedRepository = Join-Path $PSScriptRoot 'LOCAL_REPOSITORY.txt'
        if (Test-Path -LiteralPath $savedRepository -PathType Leaf) {
            $RepoPath = (Get-Content -LiteralPath $savedRepository -Raw -Encoding UTF8).Trim()
        }
    }
    if (!$RepoPath) {
        $candidates = @(
            (Join-Path (Split-Path $PSScriptRoot -Parent) 'HGV-reference-screening'),
            (Join-Path (Split-Path $PSScriptRoot -Parent) 'HGV-reference-screening_GitHub_upload_bundle/HGV-reference-screening')
        )
        foreach ($candidate in $candidates) {
            if (Test-Path -LiteralPath (Join-Path $candidate '.git')) { $RepoPath = $candidate; break }
        }
        if (!$RepoPath) { $RepoPath = Read-Host 'Enter the local HGV-reference-screening Git repository path' }
    }
    if ($RepoPath -match 'https?://|git@github\.com') {
        throw 'A local folder is required, not a GitHub web URL. Use the one-click launcher or the path saved in LOCAL_REPOSITORY.txt.'
    }
    $script:repo = [IO.Path]::GetFullPath($RepoPath.Trim().Trim('"')).TrimEnd('\','/')
    if (!(Get-Command git.exe -ErrorAction SilentlyContinue)) { throw 'Install Git for Windows first.' }
    $top = [IO.Path]::GetFullPath((Git -Arguments @('rev-parse','--show-toplevel'))).TrimEnd('\','/')
    if ($top -ne $repo) { throw 'RepoPath must be the repository root, not an inner folder.' }
    if ((Normalize-Remote (Git -Arguments @('remote','get-url','origin'))) -ne (Normalize-Remote $meta.repository_url)) {
        throw 'origin does not match the repository named in UPDATE_METADATA.json.'
    }
    Git -Arguments @('merge-base','--is-ancestor',$meta.base_commit,'HEAD') | Out-Null
    Verify-Payload
    $plan = @(Get-Plan)
    $counts = ($plan | Group-Object state | ForEach-Object { "$($_.Name)=$($_.Count)" }) -join ', '
    Write-Host "Update $($meta.update_id) | $Mode | $counts"
    Write-Host "Repository: $repo"
    if ($Mode -eq 'Plan') {
        Write-Host 'PASS: payload verified, all target paths checked; repository unchanged.'
        exit 0
    }
    if ($Mode -eq 'Apply') {
        if ((Git -Arguments @('diff','--cached','--name-only')).Trim()) { throw 'Git index is not empty. Commit or unstage your existing changes first.' }
        Git -Arguments @('config','--local','core.longpaths','true') | Out-Null
        $reportDir = New-ReportDir
        $plan | Select-Object path,state | Export-Csv -LiteralPath (Join-Path $reportDir 'before.tsv') -Delimiter "`t" -NoTypeInformation -Encoding UTF8
        foreach ($entry in $plan) {
            if ($entry.state -eq 'UNCHANGED') { continue }
            $source = Safe-Path $payload $entry.path
            # Recheck immediately before each write; interrupted applies can be resumed safely.
            if ($entry.state -eq 'MODIFY') {
                if (!(Old-Matches $entry.destination $entry.row)) { throw "File changed during apply: $($entry.path)" }
                $backup = Safe-Path $reportDir ('backup/' + $entry.path)
                [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($backup)) | Out-Null
                [IO.File]::Copy($entry.destination,$backup,$false)
            } elseif (Test-Path -LiteralPath $entry.destination) { throw "File appeared during apply: $($entry.path)" }
            [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($entry.destination)) | Out-Null
            [IO.File]::Copy($source,$entry.destination,($entry.state -eq 'MODIFY'))
        }
        Verify-Installed
        [IO.File]::Copy($manifest,(Join-Path $reportDir 'installed_manifest.tsv'),$false)
        Write-Host "PASS: installed files match SHA256; backup/audit: $reportDir"
        Write-Host 'No commit or push was performed.'
        exit 0
    }
    Verify-Installed
    if ($Mode -eq 'Verify') {
        Write-Host 'PASS: all installed files match UPDATE_MANIFEST.tsv.'
        Write-Host ('Current local commit: ' + (Git -Arguments @('rev-parse','HEAD')))
        exit 0
    }
    # Publish is explicitly invoked by the operator. Never force-push or commit unrelated paths.
    if ((Git -Arguments @('branch','--show-current')) -ne 'main') { throw 'Publish requires the main branch.' }
    if ((Git -Arguments @('diff','--cached','--name-only')).Trim()) { throw 'Git index contains existing staged changes; publication stopped.' }
    $allowed = @{}
    foreach ($row in $rows) { $allowed[$row.path] = $true }
    $changes = @( (Git -Arguments @('-c','core.quotepath=false','diff','--name-only')).Split("`n") | Where-Object { $_ } )
    foreach ($path in $changes) { if (!$allowed.ContainsKey($path)) { throw "Unrelated tracked change: $path" } }
    Git -Arguments @('fetch','origin','main') | Out-Null
    $remote = Git -Arguments @('rev-parse','origin/main')
    $head = Git -Arguments @('rev-parse','HEAD')
    $message = 'HGV incremental update ' + $meta.update_id + ': expose original workflows and frozen assets'
    if ($meta.PSObject.Properties.Name -contains 'commit_message' -and $meta.commit_message) { $message = [string]$meta.commit_message }
    if ($head -ne $remote) {
        # A previous push might have failed after this exact update was committed.
        $subject = Git -Arguments @('log','-1','--format=%s')
        $parent = Git -Arguments @('rev-parse','HEAD^')
        $commitPaths = (Git -Arguments @('-c','core.quotepath=false','diff-tree','--no-commit-id','--name-only','-r','HEAD')).Split("`n")
        if ($subject -ne $message -or $parent -ne $remote) { throw 'Local and remote main differ. Reconcile them normally before publication; no force push is used.' }
        foreach ($path in $commitPaths) { if ($path -and !$allowed.ContainsKey($path)) { throw "Unrelated path in pending commit: $path" } }
        if ($changes.Count -gt 0) { throw 'Worktree differs from the pending update commit.' }
    } else {
        $reportDir = New-ReportDir
        $pathFile = Join-Path $reportDir 'commit_paths.nul'
        [IO.File]::WriteAllText($pathFile, (($rows.path -join [char]0) + [char]0), $utf8)
        Git -Arguments @('--literal-pathspecs','add','-f',('--pathspec-from-file=' + $pathFile),'--pathspec-file-nul') | Out-Null
        $staged = Git -Arguments @('diff','--cached','--name-only')
        if ($staged.Trim()) { Git -Arguments @('commit','-m',$message) | Write-Host }
    }
    Verify-Commit
    Git -Arguments @('push','origin','HEAD:refs/heads/main') | Write-Host
    $head = Git -Arguments @('rev-parse','HEAD')
    $remoteLine = Git -Arguments @('ls-remote','origin','refs/heads/main')
    $published = ($remoteLine -split '\s+')[0]
    if ($published -ne $head) { throw "Push verification failed: remote=$published local=$head" }
    Write-Host "PASS: remote main=$published; all manifest files match committed blobs."
    exit 0
} catch {
    Write-Host ('STOP: ' + $_.Exception.Message) -ForegroundColor Red
    Write-Host 'No destructive rollback was attempted. Resolve the reported conflict and rerun.'
    exit 1
}
