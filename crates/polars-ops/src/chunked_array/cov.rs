use num_traits::AsPrimitive;
use polars_compute::var_cov::{CovState, PearsonState, WeightedPearsonState};
use polars_core::prelude::*;
use polars_core::utils::{align_chunks_binary, align_chunks_ternary};

/// Compute the covariance between two columns.
pub fn cov<T>(a: &ChunkedArray<T>, b: &ChunkedArray<T>, ddof: u8) -> Option<f64>
where
    T: PolarsNumericType,
    T::Native: AsPrimitive<f64>,
    ChunkedArray<T>: ChunkVar,
{
    let (a, b) = align_chunks_binary(a, b);
    let mut out = CovState::default();
    for (a, b) in a.downcast_iter().zip(b.downcast_iter()) {
        out.combine(&polars_compute::var_cov::cov(a, b))
    }
    out.finalize(ddof)
}

/// Compute the pearson correlation between two columns.
pub fn pearson_corr<T>(a: &ChunkedArray<T>, b: &ChunkedArray<T>) -> Option<f64>
where
    T: PolarsNumericType,
    T::Native: AsPrimitive<f64>,
    ChunkedArray<T>: ChunkVar,
{
    let (a, b) = align_chunks_binary(a, b);
    let mut out = PearsonState::default();
    for (a, b) in a.downcast_iter().zip(b.downcast_iter()) {
        out.combine(&polars_compute::var_cov::pearson_corr(a, b))
    }
    Some(out.finalize())
}

/// Compute the weighted covariance between two columns with the third column as weights.
pub fn weighted_pearson_corr<T>(
    a: &ChunkedArray<T>,
    b: &ChunkedArray<T>,
    weights: &ChunkedArray<T>,
) -> Option<f64>
where
    T: PolarsNumericType,
    T::Native: AsPrimitive<f64>,
    ChunkedArray<T>: ChunkVar,
{
    let (a, b, weights) = align_chunks_ternary(a, b, weights);
    let mut out = WeightedPearsonState::default();
    for ((a, b), weights) in a
        .downcast_iter()
        .zip(b.downcast_iter())
        .zip(weights.downcast_iter())
    {
        out.combine(&polars_compute::var_cov::weighted_pearson_corr(
            a, b, weights,
        ))
    }
    Some(out.finalize())
}
