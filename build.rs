//! Build script: Windows LibRaw integration and linking.
//!
//! On Windows we:
//! - Help the linker find LibRaw's dependencies (zlib, lcms2, jasper) via vcpkg or LIBRAW_DIR.
//! - Rely on [patch] in Cargo.toml for libraw-rs-sys to use a Windows-safe build (no -pthread on MSVC).

#[cfg(windows)]
use std::path::PathBuf;

fn main() {
    #[cfg(windows)]
    windows_libraw_setup();

    // Re-run if these env vars change (for development)
    if let Ok(dir) = std::env::var("LIBRAW_DIR") {
        println!("cargo:rerun-if-env-changed=LIBRAW_DIR");
        let _ = dir;
    }
    if let Ok(root) = std::env::var("VCPKG_ROOT") {
        println!("cargo:rerun-if-env-changed=VCPKG_ROOT");
        let _ = root;
    }
}

#[cfg(windows)]
fn windows_libraw_setup() {
    use std::env;
    use std::path::PathBuf;

    // 1) Prefer LIBRAW_DIR if set (user-provided SDK or vcpkg install root)
    if let Ok(dir) = env::var("LIBRAW_DIR") {
        let root = PathBuf::from(&dir);
        let lib_dir = root.join("lib");
        let lib_dir_alt = root.join("x64-windows").join("lib");
        if lib_dir.exists() {
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
        }
        if lib_dir_alt.exists() {
            println!("cargo:rustc-link-search=native={}", lib_dir_alt.display());
        }
        // Link LibRaw's common dependencies (libraw-rs-sys builds static raw; that lib may need these)
        println!("cargo:rustc-link-lib=zlib");
        println!("cargo:rustc-link-lib=lcms2");
        println!("cargo:rustc-link-lib=jasper");
        return;
    }

    // 2) Try vcpkg: VCPKG_ROOT or default install location
    let (vcpkg_root, triplet) = vcpkg_root_and_triplet();
    if let Some(root) = vcpkg_root {
        let lib_dir = root.join("installed").join(&triplet).join("lib");
        if lib_dir.exists() {
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
            // LibRaw's dependencies when using vendored build still need to be linked
            println!("cargo:rustc-link-lib=zlib");
            println!("cargo:rustc-link-lib=lcms2");
            println!("cargo:rustc-link-lib=jasper");
        } else {
            eprintln!(
                "cargo:warning=LibRaw (Windows): vcpkg lib dir not found: {}",
                lib_dir.display()
            );
            eprintln!(
                "cargo:warning=Install with: vcpkg install libraw (and run from a vcpkg/VS environment)"
            );
        }
        return;
    }

    // 3) MSYS2 / MinGW: common prefixes
    let msys2_roots: [Option<PathBuf>; 3] = [
        env::var("MSYS2_HOME").ok().map(PathBuf::from),
        env::var("MINGW_PREFIX").ok().map(|p| {
            let pb = PathBuf::from(&p);
            pb.parent()
                .and_then(|path| path.parent())
                .map(|path| path.to_path_buf())
                .unwrap_or(pb)
        }),
        None,
    ];
    for maybe_root in msys2_roots.iter().flatten() {
        let root = maybe_root.clone();
        let lib = root.join("lib");
        if lib.exists() {
            println!("cargo:rustc-link-search=native={}", lib.display());
            println!("cargo:rustc-link-lib=z");
            println!("cargo:rustc-link-lib=lcms2");
            println!("cargo:rustc-link-lib=jasper");
            return;
        }
    }

    // 4) If we're building with MSVC and vcpkg is not set, suggest it
    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("msvc") {
        eprintln!(
            "cargo:warning=LibRaw (Windows): No LIBRAW_DIR or vcpkg found. \
             Install LibRaw deps with: vcpkg install libraw"
        );
    }
}

#[cfg(windows)]
fn vcpkg_root_and_triplet() -> (Option<PathBuf>, String) {
    let root = std::env::var("VCPKG_ROOT")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            let home = std::env::var("USERPROFILE").ok()?;
            let v = PathBuf::from(home).join("vcpkg");
            if v.exists() {
                Some(v)
            } else {
                None
            }
        });

    let triplet = std::env::var("VCPKG_DEFAULT_TRIPLET").unwrap_or_else(|_| {
        let target = std::env::var("TARGET").unwrap_or_default();
        if target.contains("x86_64") {
            "x64-windows".to_string()
        } else {
            "x86-windows".to_string()
        }
    });

    (root, triplet)
}
