/// Utility functions for unit tests.
use std::ops::RangeInclusive;

use ndarray::prelude::*;
use proptest::prelude::*;

use crate::array::SystolicArray;

pub const ARR_SIZE: RangeInclusive<usize> = 1..=32;
pub type ArrItem = u32;

/// A range of activation/weight values that will not overflow [`ArrItem`].
pub const ARR_ITEM_RANGE: RangeInclusive<ArrItem> = 0..=255;

pub(crate) fn generate_weights_and_activations() -> BoxedStrategy<(Array2<ArrItem>, Array2<ArrItem>)>
{
    (ARR_SIZE, ARR_SIZE, ARR_SIZE)
        .prop_flat_map(|(in_features, out_features, batch_size)| {
            let weights = proptest::collection::vec(ARR_ITEM_RANGE, in_features * out_features)
                .prop_map(move |v| Array2::from_shape_vec((out_features, in_features), v).unwrap());
            let activations = proptest::collection::vec(ARR_ITEM_RANGE, in_features * batch_size)
                .prop_map(move |v| Array2::from_shape_vec((in_features, batch_size), v).unwrap());

            (weights, activations)
        })
        .boxed()
}

pub(crate) fn generate_array() -> BoxedStrategy<SystolicArray<ArrItem>> {
    (ARR_SIZE, ARR_SIZE)
        .prop_map(|(height, width)| SystolicArray::<ArrItem>::new(height, width).unwrap())
        .boxed()
}
