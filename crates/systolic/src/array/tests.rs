use crate::fault::{PeFaultRegister, RandomFault, RegisterHook, StuckAt, XorMaskHook};
use crate::test_utilities::{ARR_SIZE, generate_array, generate_weights_and_activations};
use proptest::prelude::*;
use rand::SeedableRng;
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

#[test]
fn weight_register_fault_corrupts_output() {
    // 2x2 array. PE(y=0, x=0) stores weights[0,0] = 4 (0b100).
    // Stuck-at-one on bit 0 changes 4 → 5 during weight loading.
    // PE(y=1, x=0) also picks up the fault via pass-through (0 → 1), but
    // activations[1] = 0 so the corrupted value does not affect the output.
    let weights = array![[4u32, 0], [0, 0]];
    let activations = array![[1u32], [0]];

    let hook = RegisterHook {
        target: Index2 { x: 0, y: 0 },
        register: PeFaultRegister::Weight,
        bit_index: 0,
        stuck_at: StuckAt::One,
    };

    let mut sa = SystolicArray::<u32>::new(2, 2).unwrap().with_hook(hook);
    sa.set_weights(&weights);
    let result = sa.run(&activations);

    assert_eq!(result, array![[5u32], [0]]);
}

#[test]
fn weight_fault_propagates_to_lower_pes() {
    // 3x1 array. Fault is at PE(y=1, x=0): bit 0 stuck at one.
    // All original weights are zero, activations are all one.
    //
    // Sequential loading (column x=0):
    //   y=0: passes through PE(0) only - no fault, stored weight = 0
    //   y=1: passes through PE(0) then PE(1) - fault fires at PE(1), stored weight = 1
    //   y=2: passes through PE(0), PE(1), PE(2) - corrupted at PE(1), stored weight = 1
    //
    // Output = 0*1 + 1*1 + 1*1 = 2. Without sequential propagation it would
    // be 1 (only PE(1) directly corrupted, PE(2) would stay 0).
    let weights = array![[0u32, 0, 0]];
    let activations = array![[1u32], [1], [1]];

    let hook = RegisterHook {
        target: Index2 { x: 0, y: 1 },
        register: PeFaultRegister::Weight,
        bit_index: 0,
        stuck_at: StuckAt::One,
    };

    let mut sa = SystolicArray::<u32>::new(3, 1)
        .expect("valid dimensions")
        .with_hook(hook);
    sa.set_weights(&weights);
    let result = sa.run(&activations);

    assert_eq!(result, array![[2u32]]);
}

#[test]
fn with_hook_rederives_weights() {
    // Load weights under a fault, then swap to NoFault via with_hook.
    // The output should reflect the clean weights without reloading.
    let weights = array![[4u32]];
    let activations = array![[1u32]];

    let hook = RegisterHook {
        target: Index2 { x: 0, y: 0 },
        register: PeFaultRegister::Weight,
        bit_index: 0,
        stuck_at: StuckAt::One,
    };

    let mut sa = SystolicArray::<u32>::new(1, 1).unwrap().with_hook(hook);
    sa.set_weights(&weights);
    assert_eq!(sa.run(&activations), array![[5u32]]);

    let mut sa = sa.with_hook(crate::fault::NoFault);
    assert_eq!(sa.run(&activations), array![[4u32]]);
}

#[test]
fn set_hook_rederives_weights() {
    // Move a stuck-at fault from y=1 to y=2 via set_hook without reloading
    // weights. The propagation pattern should shift accordingly.
    //
    // 3x1 array, all weights zero, all activations one.
    // Fault at y=1: output = 0 + 1 + 1 = 2 (y=1 and y=2 corrupted).
    // Fault at y=2: output = 0 + 0 + 1 = 1 (only y=2 corrupted).
    let weights = array![[0u32, 0, 0]];
    let activations = array![[1u32], [1], [1]];

    let hook_at_y1 = RegisterHook {
        target: Index2 { x: 0, y: 1 },
        register: PeFaultRegister::Weight,
        bit_index: 0,
        stuck_at: StuckAt::One,
    };

    let mut sa = SystolicArray::<u32>::new(3, 1)
        .expect("valid dimensions")
        .with_hook(hook_at_y1);
    sa.set_weights(&weights);
    assert_eq!(sa.run(&activations), array![[2u32]]);

    sa.set_hook(RegisterHook {
        target: Index2 { x: 0, y: 2 },
        register: PeFaultRegister::Weight,
        bit_index: 0,
        stuck_at: StuckAt::One,
    });
    assert_eq!(sa.run(&activations), array![[1u32]]);
}

#[test]
fn xor_mask_fault_corrupts_multiply_add() {
    // 1x1 array. The single PE computes 2 * 3 + 0 = 6 cleanly.
    // XOR mask 1 flips bit 0: 6 (0b110) XOR 1 = 7 (0b111).
    let weights = array![[3u8]];
    let activations = array![[2u8]];

    let fault = RandomFault {
        target: Index2 { x: 0, y: 0 },
        entries: Box::from([(1u8, 1.0_f64)]),
    };
    let hook = XorMaskHook::from_fault(fault, rand::rngs::StdRng::seed_from_u64(0))
        .expect("single positive weight must produce a valid distribution");

    let mut sa = SystolicArray::<u8>::new(1, 1).unwrap().with_hook(hook);
    sa.set_weights(&weights);
    let result = sa.run(&activations);

    assert_eq!(result, array![[7u8]]);
}
