//! Export filename templates.
//!
//! Default `{stem}` matches today's `{input_stem}.tiff` behaviour.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

/// Values substituted into [`crate::options::PipelineOptions::filename_template`].
#[derive(Debug, Clone)]
pub struct FilenameContext<'a> {
    pub stem: &'a str,
    pub index: usize,
    pub date: &'a str,
    pub time: &'a str,
    pub preset: &'a str,
    pub profile: &'a str,
    pub width: u32,
    pub height: u32,
}

/// Expand a template. Unknown `{tokens}` are left as-is. Path separators and
/// other illegal filename characters are replaced with `_`.
pub fn expand(template: &str, ctx: &FilenameContext<'_>) -> String {
    let src = if template.trim().is_empty() {
        "{stem}"
    } else {
        template.trim()
    };
    let mut out = String::with_capacity(src.len() + 16);
    let mut rest = src;
    while let Some(start) = rest.find('{') {
        out.push_str(&rest[..start]);
        let after = &rest[start + 1..];
        match after.find('}') {
            Some(end) => {
                let token = &after[..end];
                out.push_str(&replace_token(token, ctx));
                rest = &after[end + 1..];
            }
            None => {
                out.push('{');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    sanitize_filename(&out)
}

fn replace_token(token: &str, ctx: &FilenameContext<'_>) -> String {
    if let Some(spec) = token.strip_prefix("index:") {
        return format_index(ctx.index, spec);
    }
    match token {
        "stem" | "name" => ctx.stem.to_string(),
        "index" => ctx.index.to_string(),
        "date" => ctx.date.to_string(),
        "time" => ctx.time.to_string(),
        "preset" => {
            if ctx.preset.is_empty() {
                "export".to_string()
            } else {
                ctx.preset.to_string()
            }
        }
        "profile" => ctx.profile.to_string(),
        "w" => ctx.width.to_string(),
        "h" => ctx.height.to_string(),
        other => format!("{{{other}}}"),
    }
}

fn format_index(index: usize, spec: &str) -> String {
    let width = spec.parse::<usize>().unwrap_or(1).max(1);
    format!("{index:0width$}")
}

fn sanitize_filename(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.chars() {
        match ch {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' | '\0' => out.push('_'),
            c if c.is_control() => out.push('_'),
            c => out.push(c),
        }
    }
    let trimmed = out.trim_matches([' ', '.']);
    if trimmed.is_empty() {
        "image".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Join `dir / stem.ext`, adding ` - 2`, ` - 3`, … when the file exists.
pub fn unique_path(dir: &Path, stem: &str, ext: &str) -> PathBuf {
    let ext = ext.trim_start_matches('.');
    let first = dir.join(format!("{stem}.{ext}"));
    if !first.exists() {
        return first;
    }
    for n in 2..10_000 {
        let candidate = dir.join(format!("{stem} - {n}.{ext}"));
        if !candidate.exists() {
            return candidate;
        }
    }
    dir.join(format!("{stem} - 10000.{ext}"))
}

/// Local calendar date / time for `{date}` and `{time}`.
pub fn local_stamp() -> Result<(String, String)> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| anyhow::anyhow!("clock: {e}"))?;
    local_stamp_from_unix(now.as_secs())
}

fn local_stamp_from_unix(unix: u64) -> Result<(String, String)> {
    // Calendar math without extra crates. UTC is good enough for filenames.
    let secs = unix as i64;
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400) as u32;
    let (y, m, d) = civil_from_days(days);
    let hh = tod / 3600;
    let mm = (tod % 3600) / 60;
    let ss = tod % 60;
    Ok((
        format!("{y:04}-{m:02}-{d:02}"),
        format!("{hh:02}{mm:02}{ss:02}"),
    ))
}

/// Howard Hinnant civil-from-days (proleptic Gregorian).
fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m, d)
}

pub fn require_template(template: &str) -> Result<()> {
    if expand(template, &FilenameContext {
        stem: "frame",
        index: 1,
        date: "2026-01-01",
        time: "000000",
        preset: "web",
        profile: "sRGB",
        width: 100,
        height: 100,
    })
    .is_empty()
    {
        bail!("filename template produced an empty name");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> FilenameContext<'static> {
        FilenameContext {
            stem: "frame_001",
            index: 7,
            date: "2026-08-20",
            time: "094500",
            preset: "web",
            profile: "sRGB",
            width: 4000,
            height: 3000,
        }
    }

    #[test]
    fn default_stem_matches_today() {
        assert_eq!(expand("{stem}", &ctx()), "frame_001");
        assert_eq!(expand("", &ctx()), "frame_001");
    }

    #[test]
    fn index_zero_pad() {
        assert_eq!(expand("{index:03}", &ctx()), "007");
        assert_eq!(expand("{index}", &ctx()), "7");
    }

    #[test]
    fn tokens_and_sanitize() {
        let name = expand("{date}_{preset}_{profile}_{w}x{h}", &ctx());
        assert_eq!(name, "2026-08-20_web_sRGB_4000x3000");
        assert_eq!(expand("a/b:c", &ctx()), "a_b_c");
    }

    #[test]
    fn unique_path_suffix() {
        let dir = std::env::temp_dir().join(format!(
            "oxid-fn-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let a = unique_path(&dir, "frame", "tiff");
        std::fs::write(&a, b"x").unwrap();
        let b = unique_path(&dir, "frame", "tiff");
        assert_eq!(b.file_name().unwrap(), "frame - 2.tiff");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
