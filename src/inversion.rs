//! Negative → positive inversion: output = 1.0 - input.
//!
//! Purely linear, applied per-channel to the neutralized image.

use ndarray::Array3;

/// Invert in-place: each sample becomes `1.0 - value`.
///
/// Used for C-41 negatives: after D-min, the image represents the negative;
/// inversion yields the positive (subject to a tone curve for viewing).
pub fn invert(image: &mut Array3<f32>) {
    image.mapv_inplace(|v| 1.0 - v);
}
