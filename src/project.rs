//! Multi-image project files: serialize / deserialize the image list and each
//! image's user-facing `PipelineOptions`. Paths are stored relative to the
//! project file. Runtime fields (debug, GPU, flat-field, pinned zone) are not
//! persisted.

use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::dust::ProjectDust;
use crate::options::PipelineOptions;

/// Current project schema version. Bump when the JSON shape changes in a way
/// that needs a migration in [`load_project`].
pub const PROJECT_VERSION: u32 = 1;
/// On-disk project extension (without the dot).
pub const PROJECT_EXTENSION: &str = "oxidProj";
/// Previous project extension; still accepted when opening files.
pub const PROJECT_EXTENSION_LEGACY: &str = "c41proj";

/// Export format stored per image in a project file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ProjectExportFormat {
    #[default]
    Tiff16,
    Tiff32,
    Exr,
    Jpeg,
    ExrAces2065,
}

/// One image entry in a project file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectImage {
    pub path: PathBuf,
    #[serde(default)]
    pub export_format: ProjectExportFormat,
    #[serde(default)]
    pub options: PipelineOptions,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dust: Option<ProjectDust>,
}

/// On-disk project document.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ProjectFile {
    pub version: u32,
    pub images: Vec<ProjectImage>,
}

impl Default for ProjectFile {
    fn default() -> Self {
        Self {
            version: PROJECT_VERSION,
            images: Vec::new(),
        }
    }
}

/// Result of [`load_project`]: existing images with resolved paths, plus any
/// entries whose files were missing.
#[derive(Debug, Clone)]
pub struct LoadedProject {
    pub images: Vec<ProjectImage>,
    pub missing: Vec<PathBuf>,
}

/// Write a pretty-printed project JSON. Image and LUT paths are stored relative
/// to `path`'s parent directory when possible.
pub fn save_project(images: &[ProjectImage], path: &Path) -> anyhow::Result<()> {
    let project_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut file = ProjectFile {
        version: PROJECT_VERSION,
        images: images
            .iter()
            .map(|img| {
                let mut img = img.clone();
                img.path = relativize(&img.path, project_dir);
                img.options.output_lut_cube = img
                    .options
                    .output_lut_cube
                    .as_ref()
                    .map(|p| relativize(p, project_dir));
                img.options.lut3d_path = img
                    .options
                    .lut3d_path
                    .as_ref()
                    .map(|p| relativize(p, project_dir));
                img
            })
            .collect(),
    };
    file.version = PROJECT_VERSION;
    let json = serde_json::to_string_pretty(&file)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Load a project from JSON. Relative paths are resolved against the project
/// file's parent. Missing image files are omitted from `images` and listed in
/// `missing`. Unknown fields are ignored; missing option fields use defaults.
pub fn load_project(path: &Path) -> anyhow::Result<LoadedProject> {
    let text = std::fs::read_to_string(path)?;
    let file: ProjectFile =
        serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("Invalid project JSON: {e}"))?;
    let project_dir = path.parent().unwrap_or_else(|| Path::new("."));

    let mut images = Vec::new();
    let mut missing = Vec::new();
    for mut img in file.images {
        img.path = resolve_project_path(&img.path, project_dir);
        img.options.output_lut_cube = img
            .options
            .output_lut_cube
            .as_ref()
            .map(|p| resolve_project_path(p, project_dir));
        img.options.lut3d_path = img
            .options
            .lut3d_path
            .as_ref()
            .map(|p| resolve_project_path(p, project_dir));
        if img.path.exists() {
            images.push(img);
        } else {
            missing.push(img.path);
        }
    }
    Ok(LoadedProject { images, missing })
}

/// Convert `path` to a path relative to `base` when they share a root.
/// Falls back to an absolute path if a relative link cannot be formed.
pub fn relativize(path: &Path, base: &Path) -> PathBuf {
    let path_abs = normalize_abs(path, base);
    let base_abs = normalize_abs(base, base);

    let path_comps: Vec<Component<'_>> = path_abs.components().collect();
    let base_comps: Vec<Component<'_>> = base_abs.components().collect();

    let mut common = 0;
    while common < path_comps.len()
        && common < base_comps.len()
        && path_comps[common] == base_comps[common]
    {
        common += 1;
    }

    if common == 0 {
        return path_abs;
    }

    let mut rel = PathBuf::new();
    for _ in common..base_comps.len() {
        rel.push("..");
    }
    for c in &path_comps[common..] {
        rel.push(*c);
    }
    if rel.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        rel
    }
}

/// Resolve a stored project path against the project directory.
pub fn resolve_project_path(stored: &Path, project_dir: &Path) -> PathBuf {
    if stored.is_absolute() {
        stored.to_path_buf()
    } else {
        normalize_abs(&project_dir.join(stored), project_dir)
    }
}

fn normalize_abs(path: &Path, base: &Path) -> PathBuf {
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    };
    let mut out = PathBuf::new();
    for c in joined.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    out.push(c);
                }
            }
            other => out.push(other),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::{DminMode, Rect};
    use crate::options::{OutputStage, WbMode};

    fn temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "c41_project_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn roundtrip_preserves_image_settings() {
        let dir = temp_dir();
        let img_path = dir.join("frame.arw");
        std::fs::write(&img_path, b"not-a-real-raw").unwrap();

        let mut opts = PipelineOptions::default();
        opts.apply_crop = true;
        opts.crop_rect = Some(Rect {
            x: 10,
            y: 20,
            width: 100,
            height: 80,
        });
        opts.dmin_mode = DminMode::Fixed;
        opts.dmin_fixed = Some((0.2, 0.1, 0.05));
        opts.rotation_degrees = 90;
        opts.curve_offset = 0.12;
        opts.saturation = 1.25;
        opts.wb_mode = WbMode::Picker;
        opts.wb_r = 1.08;
        opts.output_stage = OutputStage::FilmPrint;
        opts.debug_pipeline_step = 3;
        opts.use_gpu = true;
        opts.pinned_zone = Some((0.1, 0.2, 0.3, 0.4));

        let project_path = dir.join("roll.oxidProj");
        save_project(
            &[ProjectImage {
                path: img_path.clone(),
                export_format: ProjectExportFormat::Jpeg,
                options: opts,
                dust: None,
            }],
            &project_path,
        )
        .unwrap();

        let loaded = load_project(&project_path).unwrap();
        assert!(loaded.missing.is_empty());
        assert_eq!(loaded.images.len(), 1);
        let img = &loaded.images[0];
        assert_eq!(img.path, img_path);
        assert_eq!(img.export_format, ProjectExportFormat::Jpeg);
        assert!(img.options.apply_crop);
        assert_eq!(img.options.crop_rect.unwrap().width, 100);
        assert_eq!(img.options.dmin_mode, DminMode::Fixed);
        assert_eq!(img.options.dmin_fixed, Some((0.2, 0.1, 0.05)));
        assert_eq!(img.options.rotation_degrees, 90);
        assert_eq!(img.options.curve_offset, 0.12);
        assert_eq!(img.options.saturation, 1.25);
        assert_eq!(img.options.wb_mode, WbMode::Picker);
        assert_eq!(img.options.wb_r, 1.08);
        assert_eq!(img.options.output_stage, OutputStage::FilmPrint);
        assert_eq!(img.options.debug_pipeline_step, 6);
        assert!(!img.options.use_gpu);
        assert!(img.options.pinned_zone.is_none());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_stores_paths_relative_to_project() {
        let dir = temp_dir();
        let photos = dir.join("photos");
        std::fs::create_dir_all(&photos).unwrap();
        let img_path = photos.join("DSC001.ARW");
        std::fs::write(&img_path, b"raw").unwrap();
        let lut_path = dir.join("look.cube");
        std::fs::write(&lut_path, b"lut").unwrap();

        let mut opts = PipelineOptions::default();
        opts.output_lut_cube = Some(lut_path.clone());

        let project_path = dir.join("job.oxidProj");
        save_project(
            &[ProjectImage {
                path: img_path,
                export_format: ProjectExportFormat::Tiff16,
                options: opts,
                dust: None,
            }],
            &project_path,
        )
        .unwrap();

        let text = std::fs::read_to_string(&project_path).unwrap();
        let file: ProjectFile = serde_json::from_str(&text).unwrap();
        assert_eq!(file.images[0].path, PathBuf::from("photos/DSC001.ARW"));
        assert_eq!(
            file.images[0].options.output_lut_cube.as_deref(),
            Some(Path::new("look.cube"))
        );

        let loaded = load_project(&project_path).unwrap();
        assert_eq!(loaded.images[0].path, photos.join("DSC001.ARW"));
        assert_eq!(
            loaded.images[0].options.output_lut_cube.as_deref(),
            Some(lut_path.as_path())
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn dust_strokes_roundtrip() {
        use crate::dust::{DustStroke, DustTool, ProjectDust};

        let dir = temp_dir();
        let img_path = dir.join("frame.arw");
        std::fs::write(&img_path, b"not-a-real-raw").unwrap();

        let project_path = dir.join("dust.oxidProj");
        save_project(
            &[ProjectImage {
                path: img_path.clone(),
                export_format: ProjectExportFormat::Tiff16,
                options: PipelineOptions::default(),
                dust: Some(ProjectDust {
                    reference_size: (800, 600),
                    strokes: vec![DustStroke {
                        tool: DustTool::Pen,
                        radius: 6.0,
                        points: vec![(10.0, 20.0), (12.0, 22.0)],
                    }],
                }),
            }],
            &project_path,
        )
        .unwrap();

        let loaded = load_project(&project_path).unwrap();
        let dust = loaded.images[0].dust.as_ref().unwrap();
        assert_eq!(dust.reference_size, (800, 600));
        assert_eq!(dust.strokes.len(), 1);
        assert_eq!(dust.strokes[0].tool, DustTool::Pen);
        assert_eq!(dust.strokes[0].points.len(), 2);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_fields_use_defaults() {
        let dir = temp_dir();
        let img_path = dir.join("a.arw");
        std::fs::write(&img_path, b"raw").unwrap();
        let project_path = dir.join("partial.oxidProj");
        std::fs::write(
            &project_path,
            r#"{"version":1,"images":[{"path":"a.arw"}]}"#,
        )
        .unwrap();

        let loaded = load_project(&project_path).unwrap();
        assert_eq!(loaded.images.len(), 1);
        let defaults = PipelineOptions::default();
        assert_eq!(loaded.images[0].options.curve_gamma, defaults.curve_gamma);
        assert_eq!(loaded.images[0].options.wb_mode, defaults.wb_mode);
        assert_eq!(loaded.images[0].export_format, ProjectExportFormat::Tiff16);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_image_is_reported() {
        let dir = temp_dir();
        let project_path = dir.join("gone.oxidProj");
        std::fs::write(
            &project_path,
            r#"{"version":1,"images":[{"path":"missing.arw"}]}"#,
        )
        .unwrap();

        let loaded = load_project(&project_path).unwrap();
        assert!(loaded.images.is_empty());
        assert_eq!(loaded.missing.len(), 1);
        assert!(loaded.missing[0].ends_with("missing.arw"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn relativize_same_dir_is_filename() {
        let base = Path::new("/rolls/summer");
        let path = Path::new("/rolls/summer/DSC001.ARW");
        assert_eq!(relativize(path, base), PathBuf::from("DSC001.ARW"));
    }

    #[test]
    fn resolve_keeps_absolute_paths() {
        let abs = PathBuf::from("/elsewhere/file.arw");
        assert_eq!(resolve_project_path(&abs, Path::new("/rolls/summer")), abs);
    }
}
