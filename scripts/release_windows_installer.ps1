# Build a Windows installer for Oxid (Inno Setup 6).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/release_windows_installer.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/release_windows_installer.ps1 -Version 0.1.1
#
# Skip compiling the app (reuse an existing release exe):
#   ... -SkipBuild
#
# Default toolchain is MSVC so the installer does not need MinGW DLLs.

[CmdletBinding()]
param(
    [string]$Version,
    [string]$AppName = "Oxid",
    [string]$CargoBin = "Oxid",
    [string]$CargoFeatures = "gui,gpu",
    [string]$Toolchain = "stable-x86_64-pc-windows-msvc",
    [string]$Target = "x86_64-pc-windows-msvc",
    [string]$IsccPath,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$CargoToml = Join-Path $ProjectDir "Cargo.toml"
$LogoPng = Join-Path $ProjectDir "src\img\logo.png"
$LicenseSrc = Join-Path $ProjectDir "LICENSE"
$IssScript = Join-Path $ScriptDir "oxid.iss"
$BuildDir = Join-Path $ProjectDir "build"
$DistDir = Join-Path $BuildDir "dist"
$IconPath = Join-Path $BuildDir "Oxid.ico"

function Get-CargoVersion {
    $text = Get-Content -Path $CargoToml -Raw
    if ($text -match '(?m)^version\s*=\s*"([^"]+)"') {
        return $Matches[1]
    }
    throw "Could not read version from Cargo.toml"
}

function Convert-PngToIco {
    param(
        [Parameter(Mandatory = $true)][string]$PngPath,
        [Parameter(Mandatory = $true)][string]$IcoPath
    )

    $pngBytes = [System.IO.File]::ReadAllBytes($PngPath)
    if ($pngBytes.Length -lt 24 -or $pngBytes[0] -ne 0x89 -or $pngBytes[1] -ne 0x50) {
        throw "Not a PNG: $PngPath"
    }

    $width = [System.BitConverter]::ToUInt32(
        @($pngBytes[19], $pngBytes[18], $pngBytes[17], $pngBytes[16]), 0)
    $height = [System.BitConverter]::ToUInt32(
        @($pngBytes[23], $pngBytes[22], $pngBytes[21], $pngBytes[20]), 0)

    $entryWidth = if ($width -ge 256) { 0 } else { [byte]$width }
    $entryHeight = if ($height -ge 256) { 0 } else { [byte]$height }

    $ms = New-Object System.IO.MemoryStream
    $bw = New-Object System.IO.BinaryWriter $ms
    $bw.Write([uint16]0)          # reserved
    $bw.Write([uint16]1)          # type = icon
    $bw.Write([uint16]1)          # count
    $bw.Write([byte]$entryWidth)
    $bw.Write([byte]$entryHeight)
    $bw.Write([byte]0)            # color count
    $bw.Write([byte]0)            # reserved
    $bw.Write([uint16]1)          # planes
    $bw.Write([uint16]32)         # bit count
    $bw.Write([uint32]$pngBytes.Length)
    $bw.Write([uint32]22)         # image offset
    $bw.Write($pngBytes)
    $bw.Flush()
    [System.IO.File]::WriteAllBytes($IcoPath, $ms.ToArray())
    $bw.Dispose()
    $ms.Dispose()
}

function Find-Iscc {
    param([string]$Override)
    if ($Override) {
        if (-not (Test-Path $Override)) {
            throw "ISCC.exe not found: $Override"
        }
        return $Override
    }
    $cmd = Get-Command iscc -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    $candidates = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles}\Inno Setup 6\ISCC.exe",
        "${env:LOCALAPPDATA}\Programs\Inno Setup 6\ISCC.exe"
    )
    foreach ($path in $candidates) {
        if (Test-Path $path) {
            return $path
        }
    }
    throw "Inno Setup 6 is required (ISCC.exe). Install from https://jrsoftware.org/isinfo.php"
}

if (-not $Version) {
    $Version = Get-CargoVersion
}

$Iscc = Find-Iscc -Override $IsccPath
$TargetRoot = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $ProjectDir "target" }
$ExePath = Join-Path $TargetRoot "$Target\release\$CargoBin.exe"
$HostExePath = Join-Path $TargetRoot "release\$CargoBin.exe"
$OutputBase = "$AppName-$Version-setup"

Write-Host "==> $AppName $Version -> $OutputBase.exe"

New-Item -ItemType Directory -Force -Path $BuildDir, $DistDir | Out-Null

if (-not $SkipBuild) {
    Write-Host "==> Building $CargoBin ($Toolchain, features=$CargoFeatures)"
    Push-Location $ProjectDir
    try {
        & rustup run $Toolchain cargo build --release --target $Target --bin $CargoBin --features $CargoFeatures
        if ($LASTEXITCODE -ne 0) {
            throw "cargo build failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

if (-not (Test-Path $ExePath) -and (Test-Path $HostExePath)) {
    $ExePath = $HostExePath
}
if (-not (Test-Path $ExePath)) {
    throw "Release binary not found: $ExePath"
}

if (-not (Test-Path $LogoPng)) {
    throw "Logo not found: $LogoPng"
}
if (-not (Test-Path $LicenseSrc)) {
    throw "LICENSE not found: $LicenseSrc"
}

Write-Host "==> Building app icon from $LogoPng"
Convert-PngToIco -PngPath $LogoPng -IcoPath $IconPath

function ConvertTo-InnoPath([string]$Path) {
    return ($Path -replace '\\', '/')
}

Write-Host "==> Compiling installer with Inno Setup"
$isccArgs = @(
    "/Qp",
    "/DAppName=$AppName",
    "/DAppVersion=$Version",
    "/DAppPublisher=$AppName",
    "/DSourceExe=$(ConvertTo-InnoPath $ExePath)",
    "/DSourceLicense=$(ConvertTo-InnoPath $LicenseSrc)",
    "/DSourceLogo=$(ConvertTo-InnoPath $LogoPng)",
    "/DSourceIcon=$(ConvertTo-InnoPath $IconPath)",
    "/DOutputDir=$(ConvertTo-InnoPath $DistDir)",
    "/DOutputBase=$OutputBase",
    $IssScript
)
& $Iscc @isccArgs
if ($LASTEXITCODE -ne 0) {
    throw "ISCC failed with exit code $LASTEXITCODE"
}

$Installer = Join-Path $DistDir "$OutputBase.exe"
if (-not (Test-Path $Installer)) {
    throw "Installer was not created: $Installer"
}

$item = Get-Item $Installer
Write-Host ""
Write-Host "Installer: $($item.FullName)"
Write-Host ("Size:      {0:N1} MB" -f ($item.Length / 1MB))
