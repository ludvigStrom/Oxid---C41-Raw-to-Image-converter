//! Score auto-crop against a hand-corrected `.oxidProj`.
//!
//! ```text
//! cargo run --release --example crop_bench -- "/path/to/cropped.oxidProj"
//! ```

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use c41_raw_tool::{
    load_project, probe_auto_crop_for_path, run_auto_crop_for_path, AutoCropResult, Rect,
};
use rayon::prelude::*;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let probe = args.iter().any(|a| a == "--probe");
    let project = PathBuf::from(
        args.iter()
            .find(|a| !a.starts_with("--"))
            .cloned()
            .expect("usage: crop_bench [--probe] <project.oxidProj>"),
    );
    let loaded = load_project(&project).expect("load project");
    if !loaded.missing.is_empty() {
        eprintln!("missing {} files", loaded.missing.len());
        for p in &loaded.missing {
            eprintln!("  {}", p.display());
        }
    }
    let jobs: Vec<_> = loaded
        .images
        .into_iter()
        .filter_map(|img| {
            let opts = img.options;
            let gt = opts.crop_rect?;
            let gt_ref = opts.crop_rect_reference_size?;
            Some((img.path, opts, gt, gt_ref))
        })
        .collect();
    let total = jobs.len();
    if probe {
        const WORST: &[&str] = &[
            "DSC00910.ARW",
            "DSC00883.ARW",
            "DSC00915.ARW",
            "DSC00914.ARW",
            "DSC00908.ARW",
            "DSC00907.ARW",
            "DSC00931.ARW",
            "DSC00880.ARW",
        ];
        for (path, opts, _, _) in &jobs {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("?");
            if !WORST.contains(&name) {
                continue;
            }
            match probe_auto_crop_for_path(path, opts) {
                Ok(s) => println!("{name}  {s}"),
                Err(e) => println!("{name}  ERR {e}"),
            }
        }
        return;
    }
    eprintln!("scoring {total} images…");
    let done = AtomicUsize::new(0);
    let t0 = Instant::now();

    let mut rows: Vec<Row> = jobs
        .par_iter()
        .map(|(path, opts, gt, gt_ref)| {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?")
                .to_string();
            let mut cb = |_m: &str, _f: f32, _l: Option<&str>| {};
            let result = run_auto_crop_for_path(path, opts, &mut cb);
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            if n % 4 == 0 || n == total {
                eprintln!("  {n}/{total}  {}", name);
            }
            match result {
                Ok(det) => score_row(name, *gt, *gt_ref, Some(det)),
                Err(e) => {
                    let mut row = score_row(name, *gt, *gt_ref, None);
                    row.err = Some(e.to_string());
                    row
                }
            }
        })
        .collect();
    rows.sort_by(|a, b| {
        a.iou
            .partial_cmp(&b.iou)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!(
        "{:<16} {:>6} {:>6} {:>6} {:>6} {:>6} {:>7} {:>8} {}",
        "file", "iou", "dL", "dT", "dR", "dB", "maxE", "conf", "surround"
    );
    for r in &rows {
        let conf = r.conf.as_deref().unwrap_or("-");
        let sur = r.surround.as_deref().unwrap_or("-");
        println!(
            "{:<16} {:6.3} {:6.3} {:6.3} {:6.3} {:6.3} {:6.3} {:>8} {}{}",
            r.name,
            r.iou,
            r.dl,
            r.dt,
            r.dr,
            r.db,
            r.max_e,
            conf,
            sur,
            r.err
                .as_ref()
                .map(|e| format!("  ERR {e}"))
                .unwrap_or_default()
        );
    }

    let n = rows.len().max(1) as f32;
    let mean_iou = rows.iter().map(|r| r.iou).sum::<f32>() / n;
    let ious: Vec<f32> = rows.iter().map(|r| r.iou).collect();
    let med = percentile(&ious, 0.5);
    let hit = |pred: fn(&Row) -> bool| rows.iter().filter(|r| pred(r)).count();
    println!();
    println!(
        "n={}  mean IoU={:.3}  median={:.3}  elapsed={:.1}s",
        rows.len(),
        mean_iou,
        med,
        t0.elapsed().as_secs_f32()
    );
    println!(
        "IoU>=0.80 {:>2}/{}  ({:.0}%)",
        hit(|r| r.iou >= 0.80),
        rows.len(),
        100.0 * hit(|r| r.iou >= 0.80) as f32 / n
    );
    println!(
        "IoU>=0.85 {:>2}/{}  ({:.0}%)",
        hit(|r| r.iou >= 0.85),
        rows.len(),
        100.0 * hit(|r| r.iou >= 0.85) as f32 / n
    );
    println!(
        "IoU>=0.90 {:>2}/{}  ({:.0}%)",
        hit(|r| r.iou >= 0.90),
        rows.len(),
        100.0 * hit(|r| r.iou >= 0.90) as f32 / n
    );
    println!(
        "max edge err<=0.04 {:>2}/{}  ({:.0}%)",
        hit(|r| r.max_e <= 0.04),
        rows.len(),
        100.0 * hit(|r| r.max_e <= 0.04) as f32 / n
    );
    println!(
        "usable (IoU>=0.85 or maxE<=0.05) {:>2}/{}  ({:.0}%)",
        hit(|r| r.iou >= 0.85 || r.max_e <= 0.05),
        rows.len(),
        100.0 * hit(|r| r.iou >= 0.85 || r.max_e <= 0.05) as f32 / n
    );
}

struct Row {
    name: String,
    iou: f32,
    dl: f32,
    dt: f32,
    dr: f32,
    db: f32,
    max_e: f32,
    conf: Option<String>,
    surround: Option<String>,
    err: Option<String>,
}

fn score_row(name: String, gt: Rect, gt_ref: (u32, u32), det: Option<AutoCropResult>) -> Row {
    let g = norm(gt, gt_ref);
    let (iou, dl, dt, dr, db, max_e, conf, surround) = if let Some(d) = det {
        let p = norm(d.rect, d.reference_size);
        let iou = iou_norm(g, p);
        let dl = p.0 - g.0;
        let dt = p.1 - g.1;
        let dr = p.2 - g.2;
        let db = p.3 - g.3;
        let max_e = dl.abs().max(dt.abs()).max(dr.abs()).max(db.abs());
        (
            iou,
            dl,
            dt,
            dr,
            db,
            max_e,
            Some(format!("{:?}", d.confidence)),
            Some(format!("{:?}", d.surround)),
        )
    } else {
        (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, None, None)
    };
    Row {
        name,
        iou,
        dl,
        dt,
        dr,
        db,
        max_e,
        conf,
        surround,
        err: None,
    }
}

fn norm(r: Rect, (w, h): (u32, u32)) -> (f32, f32, f32, f32) {
    let w = w.max(1) as f32;
    let h = h.max(1) as f32;
    (
        r.x as f32 / w,
        r.y as f32 / h,
        (r.x + r.width) as f32 / w,
        (r.y + r.height) as f32 / h,
    )
}

fn iou_norm(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let x0 = a.0.max(b.0);
    let y0 = a.1.max(b.1);
    let x1 = a.2.min(b.2);
    let y1 = a.3.min(b.3);
    let inter = (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
    let aa = (a.2 - a.0).max(0.0) * (a.3 - a.1).max(0.0);
    let ba = (b.2 - b.0).max(0.0) * (b.3 - b.1).max(0.0);
    let union = aa + ba - inter;
    if union <= 1e-6 {
        0.0
    } else {
        inter / union
    }
}

fn percentile(values: &[f32], p: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut s = values.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((s.len() - 1) as f32 * p).round() as usize;
    s[idx.min(s.len() - 1)]
}
