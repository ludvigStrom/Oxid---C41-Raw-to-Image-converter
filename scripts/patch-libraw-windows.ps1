# Create vendor/libraw-rs-sys from the Cargo registry and patch build.rs for Windows/MSVC.
# Run once before building. Requires: PowerShell 5.1+
#
# What it fixes in libraw-rs-sys build.rs:
#   - Removes -pthread and -static flags on MSVC (cl.exe doesn't understand them)
#   - Defines LIBRAW_BUILDLIB so the C API isn't marked dllimport (fixes C2491)
#   - Adds /EHsc for C++ exception handling on MSVC

$ErrorActionPreference = "Stop"

# Locate libraw-rs-sys in the Cargo registry
$registry = Join-Path $env:USERPROFILE ".cargo\registry\src"
$srcGlob = Join-Path $registry "index.crates.io-*\libraw-rs-sys-*"
$found = Get-Item $srcGlob -ErrorAction SilentlyContinue
if (-not $found -or $found.Count -eq 0) {
    Write-Error "libraw-rs-sys not found in Cargo registry. Run 'cargo fetch' first, then re-run this script."
}
$src = $found[0].FullName

# Copy to vendor/
$projectRoot = Split-Path $PSScriptRoot -Parent
$dest = Join-Path $projectRoot "vendor\libraw-rs-sys"
if (Test-Path $dest) {
    Remove-Item -Recurse -Force $dest
}
New-Item -ItemType Directory -Path (Split-Path $dest) -Force | Out-Null
Copy-Item -Path $src -Destination $dest -Recurse -Force
Write-Host "Copied $src -> $dest"

# Rewrite the build() function's flags section with runtime TARGET detection
$buildRs = Join-Path $dest "build.rs"
$content = Get-Content $buildRs -Raw

# Replace the flags block: everything from "libraw.warnings(false);" through "libraw.compile("raw");"
$pattern = '(?s)libraw\.warnings\(false\);.*?libraw\.compile\("raw"\);'
$replacement = @'
libraw.warnings(false);
    libraw.extra_warnings(false);

    let target = env::var("TARGET").unwrap_or_default();
    let is_msvc = target.contains("msvc");

    if !is_msvc {
        libraw.flag_if_supported("-Wno-format-truncation");
        libraw.flag_if_supported("-Wno-unused-result");
        libraw.flag_if_supported("-Wno-format-overflow");
        libraw.flag("-pthread");
        libraw.static_flag(true);
    } else {
        libraw.define("LIBRAW_BUILDLIB", None);
        libraw.flag("/EHsc");
    }

    libraw.compile("raw");
'@

if ($content -match $pattern) {
    $content = [regex]::Replace($content, $pattern, $replacement)
    Set-Content -Path $buildRs -Value $content -NoNewline
    Write-Host "Patched vendor\libraw-rs-sys\build.rs for MSVC."
} else {
    Write-Warning "build.rs flags section not found (already patched?). Check vendor\libraw-rs-sys\build.rs manually."
}

Write-Host ""
Write-Host "Done. Make sure Cargo.toml contains:"
Write-Host ""
Write-Host "  [patch.crates-io]"
Write-Host '  libraw-rs-sys = { path = "vendor/libraw-rs-sys" }'
Write-Host ""
Write-Host "Then run: cargo clean; cargo build --release"
