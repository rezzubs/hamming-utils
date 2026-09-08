/// Utility functions for unit tests.
use std::ops::RangeInclusive;

use ndarray::prelude::*;
use proptest::prelude::*;

use crate::Index2;
use crate::array::SystolicArray;
use crate::fault::RegisterFault;
use crate::space::{ArrayConfig, Space};

pub const ARR_SIZE: RangeInclusive<usize> = 1..=32;
pub type ArrItem = u32;

/// A range of activation/weight values that will not overflow [`ArrItem`].
pub const ARR_ITEM_RANGE: RangeInclusive<ArrItem> = 0..=255;

/// Bit range for register faults during tests. Restricting to the lower 8 bits
/// ensures corrupted u32 values stay within ARR_ITEM_RANGE so accumulation
/// does not overflow.
pub const ARR_ITEM_BITS_FOR_FAULT: u8 = 8;

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

pub(crate) fn generate_array_with_register_fault()
-> BoxedStrategy<(SystolicArray<ArrItem>, Index2, RegisterFault)> {
    generate_array()
        .prop_flat_map(|array| {
            let nrows = array.nrows();
            let ncols = array.ncols();
            let config = ArrayConfig::new(nrows, ncols, ARR_ITEM_BITS_FOR_FAULT);

            let index_count = Index2::count(config);
            let index = (0..index_count).prop_map(move |i| Index2::from_index(i, config));

            let fault_count = RegisterFault::count(config);
            let fault = (0..fault_count).prop_map(move |i| RegisterFault::from_index(i, config));

            (Just(array), index, fault)
        })
        .boxed()
}

pub(crate) fn index(y: usize, x: usize) -> Index2 {
    Index2 {
        x: x.try_into().expect("index fits in u16"),
        y: y.try_into().expect("index fits in u16"),
    }
}
