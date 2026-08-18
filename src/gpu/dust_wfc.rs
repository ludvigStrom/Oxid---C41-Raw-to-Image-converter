//! Wave-function dust heal. Structure-flow + residual runs on CPU
//! (cheaper than the old Jacobi search). GPU entry stays so callers
//! can keep `Err → CPU` without a second code path.

use std::sync::Arc;

use ndarray::Array3;

use super::GpuContext;
use crate::dust::{prepare_dust_heal, DustHealParams, DustHealPrep, DustMask};

pub struct DustWfcPipeline {
    _ctx: Arc<GpuContext>,
}

impl DustWfcPipeline {
    pub fn new(ctx: &Arc<GpuContext>) -> Self {
        Self {
            _ctx: Arc::clone(ctx),
        }
    }

    pub fn run(
        &self,
        image: &mut Array3<f32>,
        mask: &DustMask,
        params: DustHealParams,
    ) -> anyhow::Result<()> {
        let Some(prep) = prepare_dust_heal(image, mask, params) else {
            return Ok(());
        };
        let DustHealPrep {
            tight,
            dilated,
            alpha,
            grain,
            tile,
            loosen,
            w,
            h,
        } = prep;
        crate::dust_wfc::heal_wfc(
            image, &tight, &dilated, &alpha, grain, tile, loosen, w, h,
        );
        Ok(())
    }
}
