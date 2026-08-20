//! Detect the current display ICC. Used for preview encode, not export.

use std::sync::Arc;

/// Detected monitor profile.
#[derive(Debug, Clone)]
pub struct MonitorProfile {
    pub name: String,
    pub icc: Arc<Vec<u8>>,
}

impl MonitorProfile {
    pub fn hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.icc.hash(&mut h);
        self.name.hash(&mut h);
        h.finish()
    }
}

/// Best-effort current-display ICC. `None` means “treat preview as sRGB”.
pub fn detect() -> Option<MonitorProfile> {
    #[cfg(all(target_os = "macos", feature = "gui"))]
    {
        detect_macos()
    }
    #[cfg(target_os = "windows")]
    {
        detect_windows()
    }
    #[cfg(not(any(all(target_os = "macos", feature = "gui"), target_os = "windows")))]
    {
        None
    }
}

#[cfg(all(target_os = "macos", feature = "gui"))]
fn detect_macos() -> Option<MonitorProfile> {
    detect_macos_inner().ok().flatten()
}

#[cfg(all(target_os = "macos", feature = "gui"))]
fn detect_macos_inner() -> Result<Option<MonitorProfile>, ()> {
    use objc2::rc::Retained;
    use objc2_app_kit::{NSColorSpace, NSScreen};
    use objc2_foundation::MainThreadMarker;

    let mtm = MainThreadMarker::new().ok_or(())?;
    let screen = NSScreen::mainScreen(mtm).ok_or(())?;
    let space: Retained<NSColorSpace> = unsafe { screen.colorSpace() }.ok_or(())?;
    let name = unsafe { space.localizedName() }
        .map(|s| s.to_string())
        .unwrap_or_else(|| "Display".to_string());
    let data = unsafe { space.ICCProfileData() }.ok_or(())?;
    let len = data.length();
    if len < 128 {
        return Ok(None);
    }
    let ptr = data.bytes();
    let bytes = unsafe { std::slice::from_raw_parts(ptr.as_ptr().cast::<u8>(), len) }.to_vec();
    if bytes.len() < 128 || &bytes[36..40] != b"acsp" {
        return Ok(None);
    }
    Ok(Some(MonitorProfile {
        name,
        icc: Arc::new(bytes),
    }))
}

#[cfg(target_os = "windows")]
fn detect_windows() -> Option<MonitorProfile> {
    detect_windows_inner()
}

#[cfg(target_os = "windows")]
fn detect_windows_inner() -> Option<MonitorProfile> {
    use std::os::windows::ffi::OsStringExt;
    use std::ptr;

    #[link(name = "gdi32")]
    extern "system" {
        fn CreateDCW(
            driver: *const u16,
            device: *const u16,
            output: *const u16,
            init: *const core::ffi::c_void,
        ) -> *mut core::ffi::c_void;
        fn DeleteDC(hdc: *mut core::ffi::c_void) -> i32;
        fn GetICMProfileW(hdc: *mut core::ffi::c_void, size: *mut u32, buf: *mut u16) -> i32;
    }

    let display: Vec<u16> = "DISPLAY\0".encode_utf16().collect();
    let hdc = unsafe { CreateDCW(display.as_ptr(), ptr::null(), ptr::null(), ptr::null()) };
    if hdc.is_null() {
        return None;
    }
    let mut size = 0u32;
    let _ = unsafe { GetICMProfileW(hdc, &mut size, ptr::null_mut()) };
    if size == 0 {
        unsafe { DeleteDC(hdc) };
        return None;
    }
    let mut buf = vec![0u16; size as usize];
    let ok = unsafe { GetICMProfileW(hdc, &mut size, buf.as_mut_ptr()) };
    unsafe { DeleteDC(hdc) };
    if ok == 0 {
        return None;
    }
    let path = std::ffi::OsString::from_wide(&buf[..size.saturating_sub(1) as usize]);
    let path = std::path::PathBuf::from(path);
    let bytes = std::fs::read(&path).ok()?;
    if bytes.len() < 128 || &bytes[36..40] != b"acsp" {
        return None;
    }
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Display")
        .to_string();
    Some(MonitorProfile {
        name,
        icc: Arc::new(bytes),
    })
}
