param(
    [Parameter(Mandatory = $true)][int]$Site,
    [Parameter(Mandatory = $true)][ValidateSet("head", "tail")][string]$Shard,
    [Parameter(Mandatory = $true)][int]$NEstimators,
    [Parameter(Mandatory = $true)][string]$Session,
    [Parameter(Mandatory = $true)][string]$ColabHome,
    # The 137-feature line stores its shards under a different root name, so the
    # site+estimator convention alone cannot locate them.
    [string]$ShardRootName = $null,
    [int]$Microbatch = 1024,
    [string]$ColabCli = "/home/tonykuo/.local/bin/colab",
    [ValidateSet("adc", "oauth2")][string]$Auth = "oauth2",
    [switch]$RelaxTokenScope,
    [switch]$SkipLaunch
)

# Deploys one per-site estimator-sweep shard onto an existing Colab session.
# Mirrors deploy_m5_tabpfn_colab_head.ps1, but every path comes from the shard
# manifest so the same script serves all four concurrent shards. Per-site
# feature matrices are small (12-46 MB), so only the 203 MB foundation
# checkpoint is uploaded in parts.

$ErrorActionPreference = "Stop"
$repo = "C:\Users\tonykuo\projects\lead-reproduction"
if (-not $ShardRootName) {
    $ShardRootName = "m5_tabpfn_site${Site}_context100000_n${NEstimators}"
}
$shardRoot = Join-Path $repo "data\processed\$ShardRootName"
$shardDir = Join-Path $shardRoot $Shard
$results = Join-Path $shardRoot "$Shard-results"
$foundationParts = Join-Path $repo "data\processed\m5_tabpfn_upload_parts"
$generated = Join-Path $repo ".scratch\site${Site}_${Shard}"

if ($Session -notmatch '^[A-Za-z0-9_-]+$') {
    throw "Unsafe session name: $Session"
}
if (-not (Test-Path -LiteralPath (Join-Path $shardDir "manifest.json"))) {
    throw "Missing shard manifest: $shardDir"
}
$manifest = Get-Content -LiteralPath (Join-Path $shardDir "manifest.json") -Raw | ConvertFrom-Json
$remoteRoot = $manifest.remote_root
if ($manifest.fit_state.n_estimators -ne $NEstimators) {
    throw "Manifest estimator count $($manifest.fit_state.n_estimators) != -NEstimators $NEstimators"
}

function Convert-ToWslPath([string]$WindowsPath) {
    $full = [System.IO.Path]::GetFullPath($WindowsPath)
    if ($full -notmatch '^([A-Za-z]):\\(.*)$') {
        throw "Only absolute Windows drive paths are supported: $full"
    }
    $drive = $Matches[1].ToLowerInvariant()
    $tail = $Matches[2].Replace('\', '/')
    return "/mnt/$drive/$tail"
}

function Invoke-Colab([string[]]$Arguments) {
    if ($RelaxTokenScope) {
        $output = & wsl.exe -d Ubuntu -- env "HOME=$ColabHome" "OAUTHLIB_RELAX_TOKEN_SCOPE=1" $ColabCli --auth $Auth @Arguments 2>&1
    }
    else {
        $output = & wsl.exe -d Ubuntu -- env "HOME=$ColabHome" $ColabCli --auth $Auth @Arguments 2>&1
    }
    if ($LASTEXITCODE -ne 0) {
        throw ($output -join [Environment]::NewLine)
    }
    return @($output)
}

function Invoke-RemoteScript([string]$LocalPath, [int]$SetupTimeout) {
    Invoke-Colab @(
        "exec", "-s", $Session, "-f", (Convert-ToWslPath $LocalPath),
        "--timeout", "$SetupTimeout"
    )
}

function Upload([string]$LocalPath, [string]$RemotePath) {
    if (-not (Test-Path -LiteralPath $LocalPath -PathType Leaf)) {
        throw "Missing upload input: $LocalPath"
    }
    # Colab returns transient 500/503s on upload, and the 137-feature shards push
    # 600+ MB across dozens of parts, so a single unretried failure would abort a
    # deployment that was otherwise fine. Retry transport errors with backoff;
    # a genuinely bad payload still fails after the last attempt, and the remote
    # SHA-256 check remains the real correctness gate.
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        try {
            Invoke-Colab @(
                "upload", "-s", $Session, (Convert-ToWslPath $LocalPath), $RemotePath
            ) | Out-Null
            return
        }
        catch {
            $detail = $_.Exception.Message.Replace([Environment]::NewLine, ' ')
            if ($attempt -eq 5) { throw }
            Write-Output "[deploy] upload retry $attempt for $(Split-Path $RemotePath -Leaf): $($detail.Substring(0, [Math]::Min(120, $detail.Length)))"
            Start-Sleep -Seconds (10 * $attempt)
        }
    }
}

Write-Output "[deploy] site $Site $Shard n=$NEstimators -> $Session ($remoteRoot)"

& uv run python (Join-Path $repo "scripts\build_m5_tabpfn_site_colab_scripts.py") `
    --shard-root $shardRoot --shard $Shard --query-microbatch-size $Microbatch
if ($LASTEXITCODE -ne 0) { throw "remote script generation failed" }

Invoke-RemoteScript (Join-Path $generated "create_dirs.py") 120

# The 17-feature per-site matrices are tens of MB and upload whole. The
# 137-feature ones are ~350 MB per half, where a single transfer is slow and
# fragile, so they go up in parts. The remote reassemble step already joins
# "<name>.part*" and then verifies the joined file against the manifest digest,
# so a bad split cannot pass silently.
$featuresPath = Join-Path $shardDir "features.float32.npy"
$featuresBytes = (Get-Item -LiteralPath $featuresPath).Length
# 64 MB is the only single-transfer size proven reliable here: the foundation
# checkpoint has always shipped as 64 MB parts, and the 12-46 MB shards upload
# whole without trouble. A 79 MB single upload failed on two different VMs with
# 500 then 400, so the threshold matches the proven part size rather than
# sitting above it.
$partThreshold = 64MB
$partSize = 64MB
if ($featuresBytes -gt $partThreshold) {
    $partDir = Join-Path $shardDir "upload_parts"
    New-Item -ItemType Directory -Path $partDir -Force | Out-Null
    $existing = @(Get-ChildItem -LiteralPath $partDir -Filter "features.float32.npy.part*" -File)
    if ($existing.Count -eq 0) {
        Write-Output "[deploy] splitting features ($([math]::Round($featuresBytes/1MB)) MB) into parts"
        $stream = [System.IO.File]::OpenRead($featuresPath)
        try {
            $buffer = New-Object byte[] $partSize
            $index = 0
            while (($read = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
                $partPath = Join-Path $partDir ("features.float32.npy.part{0:d3}" -f $index)
                $out = [System.IO.File]::OpenWrite($partPath)
                try { $out.Write($buffer, 0, $read) } finally { $out.Close() }
                $index++
            }
        }
        finally { $stream.Close() }
    }
    foreach ($part in Get-ChildItem -LiteralPath $partDir -Filter "features.float32.npy.part*" -File | Sort-Object Name) {
        Write-Output "[deploy] uploading $($part.Name)"
        Upload $part.FullName "$remoteRoot/$($part.Name)"
    }
}
else {
    Write-Output "[deploy] uploading features.float32.npy"
    Upload $featuresPath "$remoteRoot/features.float32.npy"
}
foreach ($name in @("metadata.npz", "model.portable.tabpfn_fit", "manifest.json")) {
    Write-Output "[deploy] uploading $name"
    Upload (Join-Path $shardDir $name) "$remoteRoot/$name"
}
foreach ($part in Get-ChildItem -LiteralPath $foundationParts -Filter "tabpfn-v3-classifier-v3_default.ckpt.part*" -File | Sort-Object Name) {
    Write-Output "[deploy] uploading $($part.Name)"
    Upload $part.FullName "$remoteRoot/$($part.Name)"
}
Upload (Join-Path $repo "scripts\run_m5_tabpfn_portable_shard.py") "$remoteRoot/run_m5_tabpfn_portable_shard.py"
Upload (Join-Path $repo "scripts\calibrate_m5_tabpfn_microbatch.py") "$remoteRoot/calibrate_m5_tabpfn_microbatch.py"

# Durable local checkpoints let a rebuilt session resume from the frontier.
$chunkDir = Join-Path $results "chunks"
if (Test-Path -LiteralPath $chunkDir) {
    foreach ($chunk in Get-ChildItem -LiteralPath $chunkDir -Filter "rows_*.npz" -File | Sort-Object Name) {
        Upload $chunk.FullName "$remoteRoot/work/chunks/$($chunk.Name)"
    }
}

Invoke-RemoteScript (Join-Path $generated "reassemble.py") 600
foreach ($name in @(
        ".scratch\install_colab_tabpfn.py",
        ".scratch\install_colab_exact_runtime.py"
    )) {
    Invoke-RemoteScript (Join-Path $repo $name) 900
}

# Runbook policy: the read-only runtime diagnostic is not a model step. A
# transient remote-exec failure here must degrade to a warning rather than
# abort a deployment whose uploads and SHA-256 checks already passed.
try {
    Invoke-RemoteScript (Join-Path $repo ".scratch\inspect_colab_hardware_subprocess.py") 300
}
catch {
    Write-Output "[deploy] runtime_inspect_warning=true detail=$($_.Exception.Message.Replace([Environment]::NewLine, ' '))"
}

if (-not $SkipLaunch) {
    Invoke-RemoteScript (Join-Path $generated "launch.py") 120
}
Write-Output "[deploy] site $Site $Shard ready"
