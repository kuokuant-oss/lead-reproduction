param(
    [string]$Session = "lead-tabpfn-tail-2",
    [string]$ColabCli = "/home/tonykuo/.local/bin/colab",
    [ValidateSet("adc", "oauth2")]
    [string]$Auth = "oauth2",
    [string]$ColabHome = "/home/tonykuo/.colab-hank"
)

$ErrorActionPreference = "Stop"
$repo = "C:\Users\tonykuo\projects\lead-reproduction"
$head = Join-Path $repo "data\processed\m5_tabpfn_distributed_context100000\head"
$results = Join-Path $repo "data\processed\m5_tabpfn_distributed_context100000\head-results"
$featureParts = Join-Path $repo "data\processed\m5_tabpfn_head_upload_parts"
$foundationParts = Join-Path $repo "data\processed\m5_tabpfn_upload_parts"
$remoteRoot = "/content/lead_tabpfn_head"
if ($Session -notmatch '^[A-Za-z0-9_-]+$') {
    throw "Unsafe session name: $Session"
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
    $output = & wsl.exe -d Ubuntu -- env "HOME=$ColabHome" $ColabCli --auth $Auth @Arguments 2>&1
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
    Invoke-Colab @(
        "upload", "-s", $Session, (Convert-ToWslPath $LocalPath), $RemotePath
    ) | Out-Null
}

Invoke-RemoteScript (Join-Path $repo "scripts\create_m5_tabpfn_colab_head_dirs.py") 120
foreach ($part in Get-ChildItem -LiteralPath $featureParts -Filter "features.float32.npy.part*" -File | Sort-Object Name) {
    Upload $part.FullName "$remoteRoot/$($part.Name)"
}
foreach ($part in Get-ChildItem -LiteralPath $foundationParts -Filter "tabpfn-v3-classifier-v3_default.ckpt.part*" -File | Sort-Object Name) {
    Upload $part.FullName "$remoteRoot/$($part.Name)"
}
foreach ($name in @("metadata.npz", "model.portable.tabpfn_fit", "manifest.json")) {
    Upload (Join-Path $head $name) "$remoteRoot/$name"
}
Upload (Join-Path $repo "scripts\run_m5_tabpfn_portable_shard.py") "$remoteRoot/run_m5_tabpfn_portable_shard.py"
foreach ($chunk in Get-ChildItem -LiteralPath (Join-Path $results "chunks") -Filter "rows_*.npz" -File | Sort-Object Name) {
    Upload $chunk.FullName "$remoteRoot/work/chunks/$($chunk.Name)"
}
Invoke-RemoteScript (Join-Path $repo "scripts\reassemble_m5_tabpfn_colab_head.py") 300
foreach ($name in @(
    ".scratch\install_colab_tabpfn.py",
    ".scratch\install_colab_exact_runtime.py",
    ".scratch\inspect_colab_python_deps.py"
)) {
    Invoke-RemoteScript (Join-Path $repo $name) 900
}
Invoke-RemoteScript (Join-Path $repo "scripts\launch_m5_tabpfn_colab_head.py") 120
