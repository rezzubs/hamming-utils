use crate::test_utilities::{ARR_SIZE, generate_array, generate_weights_and_activations};
use proptest::prelude::*;
use std::ops::Range;

use super::*;
use ndarray::array;

#[test]
fn run_shifted() {
    let mut sa = SystolicArray::<u8>::new(2, 2).unwrap();
    sa.set_weights(&array![[1, 2], [3, 4]]);

    let activations = array![[0, 1], [2, 0]];

    // See that running many times gives the same output when activations
    // are padded with zeros
    for _ in 0..100 {
        let output = sa.run_shifted(activations.clone());

        let expected_out = array![[0, 11], [5, 0]];
        assert_eq!(output, expected_out);
    }

    // Batch activations
    let activations = array![[0, 1, 1], [2, 2, 0]];
    let output = sa.run_shifted(activations.clone());
    let expected_out = array![[0, 11], [5, 11], [5, 0]];
    assert_eq!(output, expected_out);

    let activations = array![[0, 5, 6], [7, 8, 0]];
    let output = sa.run_shifted(activations);
    let expected_out = array![[0, 43], [19, 50], [22, 0]];
    assert_eq!(output, expected_out);
}

#[test]
fn test_shift_activations() {
    let activations = array![[1, 2], [3, 4]];
    let output = shift_activations(&activations);
    let expected_out = array![[0, 1, 2], [3, 4, 0]];
    assert_eq!(output, expected_out);
}

#[test]
fn test_unshift_output() {
    let raw_output = array![[0, 3], [1, 4], [2, 0]];
    let output = unshift_output(&raw_output);
    let expected_out = array![[1, 2], [3, 4]];
    assert_eq!(output, expected_out);
}

/// Split passes with at least two inputs and outputs into four quadrants,
/// each quadrant will be given its own pass.
///
/// This keeps the input/output indices the same, gaps are added to the
/// other three quadrants when appropriate.
// TODO: Use
#[allow(dead_code)]
fn partition_mapping(base_mapping: Mapping) -> Mapping {
    fn split_range(range: Range<usize>) -> (Range<usize>, Range<usize>) {
        let half_index = range.len() / 2;
        let start = range.start;
        let end = range.end;
        (start..(start + half_index), (start + half_index)..end)
    }

    let mut passes = Vec::<Pass>::new();

    for pass in base_mapping {
        if pass.activation_rows.len() == 1 || pass.output_rows.len() == 1 {
            // Cannot really split along a dimension with only one element
            passes.push(pass);
            continue;
        }

        let (a1, a2) = split_range(pass.activation_rows);
        let (o1, o2) = split_range(pass.output_rows);

        let quadrants = [
            Pass {
                activation_rows: a1.clone(),
                output_rows: o1.clone(),
                ..pass
            },
            Pass {
                activation_rows: a1.clone(),
                output_rows: o2.clone(),
                array_col_start: pass.array_col_start + o1.len(),
                ..pass
            },
            Pass {
                activation_rows: a2.clone(),
                array_row_start: pass.array_row_start + a1.len(),
                output_rows: o1.clone(),
                ..pass
            },
            Pass {
                activation_rows: a2,
                array_row_start: pass.array_row_start + a1.len(),
                output_rows: o2,
                array_col_start: pass.array_col_start + o1.len(),
            },
        ];
        passes.extend(quadrants);
    }

    let mapping = Mapping::new(passes);

    mapping
        .validate()
        .expect("the mapping translation is invalid");

    mapping
}

proptest! {
    #[test]
    fn matmul(
        (weights, activations) in generate_weights_and_activations(),
    ) {
        let expected_result = weights.dot(&activations);

        let mut sa = SystolicArray::<u32>::from_weights(&weights);

        let result = sa.run(&activations);
        assert_eq!(result, expected_result);
    }

    #[test]
    fn auto_mapping_is_valid(
        input_row_count in ARR_SIZE,
        output_row_count in ARR_SIZE,
        arr_width in ARR_SIZE,
        arr_height in ARR_SIZE
    ) {
        let array = SystolicArray::<u8>::new(arr_height, arr_width).unwrap();

        let mapping = array.auto_mapping(input_row_count, output_row_count);

        assert_eq!(mapping.validate(), Ok(()));
        assert!(array.supports_mapping(&mapping));
    }

    #[test]
    fn auto_mapping_matmul(
        mut arr in generate_array(),
        (weights, activations) in generate_weights_and_activations(),
    ) {
        let mapping = arr.auto_mapping_for(&weights);
        let result = arr.matmul(&mapping, &weights, &activations);
        let expected_result = weights.dot(&activations);
        assert_eq!(result, expected_result);
    }

    #[test]
    fn padded_mapping_matmul(
        mut arr in generate_array(),
        (weights, activations) in generate_weights_and_activations(),
    ) {
        let auto_mapping = arr.auto_mapping_for(&weights);
        let mapping = partition_mapping(auto_mapping);
        mapping.validate().unwrap();

        let result = arr.matmul(&mapping, &weights, &activations);
        let expected_result = weights.dot(&activations);
        assert_eq!(result, expected_result);
    }
}
