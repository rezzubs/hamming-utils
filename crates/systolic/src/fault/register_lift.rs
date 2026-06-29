use std::{collections::HashSet, ops::Range};

use ndarray::prelude::*;

use crate::Mapping;
use crate::fault::register::{PeFaultRegister, PeRegisterFault, RegisterFault, TargetedFault};
use crate::Index2;

/// The accumulated-fault fix-up for one output row in one pass.
///
/// The accumulator at the faulty PE holds the partial sum `P` over the
/// contributing activation rows. The fix-up replaces the clean partial sum
/// with its corrupted form: `faulty(o) = clean(o) - P + corrupt(P)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccumulatorFaultPart {
    /// The effect applies to this output row.
    pub affected_output_row: usize,
    /// The activation rows whose products had reached the faulty PE. The clean
    /// partial sum over these is subtracted from the output row and its
    /// corrupted form added back.
    ///
    /// Which rows contribute depends on where the fault sits relative to the
    /// pass's used band: within the band it is the rows at or above the fault;
    /// below the band it is all used rows; above the band nothing had
    /// accumulated, so the range is empty (`P = 0`, meaning `corrupt(0)` is
    /// added to the output row).
    pub for_activations: Range<usize>,
}

/// A register fault lifted to matrix space: an equivalent description of the
/// fault in terms of operations on the input/output matrices rather than
/// in terms of the array's physical registers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiftedRegisterFaultData {
    /// A fault in a weight register.
    ///
    /// Weights are loaded top-to-bottom, so every weight that passes through
    /// the faulty PE picks up the stuck bit. This covers the weight that settles
    /// at the fault position and every weight destined for a row below it.
    Weight {
        /// Indices into the weight matrix of the entries affected by the fault.
        affected_weights: HashSet<Index2>,
    },
    /// A fault in an activation register.
    ///
    /// The multiplication is run twice: once with clean activations and once
    /// with the affected activation rows corrupted. Output rows produced by
    /// columns to the right of the fault (those that saw the corrupted value)
    /// are taken from the faulty run; all other rows come from the clean run.
    Activation {
        /// The activation rows corrupted in the faulty run. Corrupting a full
        /// row automatically covers all batch items.
        affected_activation_rows: HashSet<usize>,
        /// The output rows taken from the faulty run rather than the clean run.
        affected_output_rows: HashSet<usize>,
    },
    /// A fault in the accumulator.
    Accumulator {
        /// One fix-up entry per pass whose column produces output at the fault
        /// position. Passes whose column does not intersect the fault have no
        /// entry.
        parts: Vec<AccumulatorFaultPart>,
    },
}

/// A register fault lifted to matrix space.
///
/// Produced by [`Mapping::lift_register_fault`]. Applying the fault via
/// [`LiftedRegisterFault::matmul`] is equivalent to running the same fault
/// through the literal systolic array, and much faster.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiftedRegisterFault {
    pub fault: RegisterFault,
    pub data: LiftedRegisterFaultData,
}

impl LiftedRegisterFault {
    /// Run a matrix multiplication with this fault applied.
    ///
    /// Produces a result identical to running the same computation through a
    /// systolic array with a `RegisterHook` for the corresponding fault.
    ///
    /// # Panics
    ///
    /// Panics if fault indices are out of bounds.
    pub fn matmul<T>(&self, mut weights: Array2<T>, activations: Array2<T>) -> Array2<T>
    where
        T: ndarray::LinalgScalar + memory::BitBuffer,
    {
        match &self.data {
            LiftedRegisterFaultData::Weight { affected_weights } => {
                for &index in affected_weights {
                    let weight = weights.get_mut(index).expect("weight index out of bounds");
                    *weight = self.fault.apply(*weight);
                }
                weights.dot(&activations)
            }
            LiftedRegisterFaultData::Activation {
                affected_activation_rows,
                affected_output_rows,
            } => {
                let mut output = weights.dot(&activations);

                let mut activations_faulty = activations.clone();
                for &row_index in affected_activation_rows {
                    activations_faulty
                        .row_mut(row_index)
                        .mapv_inplace(|v| self.fault.apply(v));
                }
                let output_faulty = weights.dot(&activations_faulty);

                for &row_index in affected_output_rows {
                    output
                        .row_mut(row_index)
                        .assign(&output_faulty.row(row_index));
                }

                output
            }
            LiftedRegisterFaultData::Accumulator { parts } => {
                let mut output = weights.dot(&activations);

                for part in parts {
                    // The corrupted partial sum is the contribution of
                    // `for_activations` to this output row. An empty range means
                    // no products reached the faulty PE, so the partial sum is
                    // zero; we state that directly rather than forming a
                    // zero-length matrix product.
                    let partial_sums = if part.for_activations.is_empty() {
                        Array1::zeros(activations.ncols())
                    } else {
                        let partial_sum_weights = weights
                            .slice(s![part.affected_output_row, part.for_activations.clone()]);
                        let partial_sum_activations =
                            activations.slice(s![part.for_activations.clone(), ..]);
                        partial_sum_weights.dot(&partial_sum_activations)
                    };

                    // The single fix-up applied to every accumulator fault:
                    // remove the clean partial sum and add back its corrupted form.
                    let output_row = output.row_mut(part.affected_output_row);
                    ndarray::Zip::from(output_row).and(&partial_sums).for_each(
                        |out_value, &partial_sum| {
                            *out_value = *out_value - partial_sum + self.fault.apply(partial_sum);
                        },
                    );
                }

                output
            }
        }
    }
}

impl Mapping {
    fn lift_weight_fault(&self, fault: RegisterFault, index: Index2) -> LiftedRegisterFault {
        let mut affected_weights = HashSet::new();

        for pass in self {
            let Some(output_row) = pass.output_row_from_array_col(index.x.into()) else {
                continue;
            };

            // All the rows at fault_y and after are affected by the fault because
            // weights are loaded from the top down.
            let Some(number_to_skip) = (index.y as usize).checked_sub(pass.array_row_start) else {
                continue;
            };
            for activation_row in pass.activation_rows.clone().skip(number_to_skip) {
                let fresh = affected_weights.insert(Index2 {
                    y: u16::try_from(output_row).expect("output_row must fit in u16"),
                    x: u16::try_from(activation_row).expect("activation_row must fit in u16"),
                });
                debug_assert!(fresh, "duplicate weight index in affected_weights");
            }
        }

        LiftedRegisterFault {
            fault,
            data: LiftedRegisterFaultData::Weight { affected_weights },
        }
    }

    fn lift_activation_fault(
        &self,
        fault: RegisterFault,
        index: Index2,
    ) -> LiftedRegisterFault {
        let mut affected_activation_rows = HashSet::<usize>::new();
        let mut affected_output_rows = HashSet::<usize>::new();

        for pass in self {
            if let Some(activation) = pass.activation_row_from_array_row(index.y.into()) {
                affected_activation_rows.insert(activation);
            }

            if let Some(output_start) = pass.output_row_from_array_col(index.x.into()) {
                affected_output_rows.extend(output_start..pass.output_rows.end);
            }
        }

        LiftedRegisterFault {
            fault,
            data: LiftedRegisterFaultData::Activation {
                affected_activation_rows,
                affected_output_rows,
            },
        }
    }

    fn lift_accumulator_fault(
        &self,
        fault: RegisterFault,
        index: Index2,
    ) -> LiftedRegisterFault {
        let mut parts = Vec::new();

        for pass in self {
            let Some(affected_output_row) = pass.output_row_from_array_col(index.x.into()) else {
                // The fault does not change the output value for this pass.
                continue;
            };

            // The partial sum at the faulty PE has accumulated the pass's used
            // rows at or above it. Intersect "rows at or above the fault" with
            // the used band: a fault above the band contributes nothing (empty
            // range), one below the band sees the whole column, and one inside
            // sees the rows down to it. Clamping handles all three uniformly.
            let used_rows = pass.range_y();
            let contributing_end =
                (usize::from(index.y) + 1).clamp(used_rows.start, used_rows.end);
            let contributing_len = contributing_end - used_rows.start;
            let for_activations =
                pass.activation_rows.start..(pass.activation_rows.start + contributing_len);

            parts.push(AccumulatorFaultPart { affected_output_row, for_activations });
        }

        LiftedRegisterFault {
            fault,
            data: LiftedRegisterFaultData::Accumulator { parts },
        }
    }

    /// Lift a targeted register fault from array space to matrix space.
    ///
    /// Returns a [`LiftedRegisterFault`] that can be applied via
    /// [`LiftedRegisterFault::matmul`] to produce the same output as running
    /// the fault through a literal systolic array, without simulating the array.
    pub fn lift_register_fault(
        &self,
        fault: &TargetedFault<PeRegisterFault>,
    ) -> LiftedRegisterFault {
        let register_fault = fault.fault.fault;
        let index = fault.target;
        match fault.fault.register {
            PeFaultRegister::Weight => self.lift_weight_fault(register_fault, index),
            PeFaultRegister::Activation => self.lift_activation_fault(register_fault, index),
            PeFaultRegister::Accumulator => self.lift_accumulator_fault(register_fault, index),
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use proptest::prelude::*;

    use crate::fault::register::{PeFaultRegister, PeRegisterFault, RegisterHook, TargetedFault};
    use crate::test_utilities::{
        ArrItem, generate_array_with_register_fault, generate_weights_and_activations,
    };

    fn make_targeted_fault(
        index: crate::Index2,
        register: PeFaultRegister,
        fault: crate::fault::RegisterFault,
    ) -> TargetedFault<PeRegisterFault> {
        TargetedFault {
            target: index,
            fault: PeRegisterFault { register, fault },
        }
    }

    fn literal_matmul(
        array: crate::array::SystolicArray<ArrItem>,
        targeted: TargetedFault<PeRegisterFault>,
        mapping: &crate::Mapping,
        weights: &Array2<ArrItem>,
        activations: &Array2<ArrItem>,
    ) -> Array2<ArrItem> {
        let hook = RegisterHook::from_fault(targeted);
        let mut faulty_array = array.with_hook(hook);
        faulty_array.matmul(mapping, weights, activations)
    }

    proptest! {
        #[test]
        fn weight_fault_matches_literal(
            (array, index, fault) in generate_array_with_register_fault(),
            (weights, activations) in generate_weights_and_activations(),
        ) {
            let mapping = array.auto_mapping_for(&weights);
            let targeted = make_targeted_fault(index, PeFaultRegister::Weight, fault);

            let expected = literal_matmul(array, targeted.clone(), &mapping, &weights, &activations);
            let result = mapping.lift_register_fault(&targeted).matmul(weights, activations);

            prop_assert_eq!(result, expected);
        }

        #[test]
        fn activation_fault_matches_literal(
            (array, index, fault) in generate_array_with_register_fault(),
            (weights, activations) in generate_weights_and_activations(),
        ) {
            let mapping = array.auto_mapping_for(&weights);
            let targeted = make_targeted_fault(index, PeFaultRegister::Activation, fault);

            let expected = literal_matmul(array, targeted.clone(), &mapping, &weights, &activations);
            let result = mapping.lift_register_fault(&targeted).matmul(weights, activations);

            prop_assert_eq!(result, expected);
        }

        #[test]
        fn accumulator_fault_matches_literal(
            (array, index, fault) in generate_array_with_register_fault(),
            (weights, activations) in generate_weights_and_activations(),
        ) {
            let mapping = array.auto_mapping_for(&weights);
            let targeted = make_targeted_fault(index, PeFaultRegister::Accumulator, fault);

            let expected = literal_matmul(array, targeted.clone(), &mapping, &weights, &activations);
            let result = mapping.lift_register_fault(&targeted).matmul(weights, activations);

            prop_assert_eq!(result, expected);
        }
    }

    /// An accumulator fault above the used band has an empty contributing range
    /// (a "faulty zero"). Auto mappings never use offsets, so the proptests
    /// above cannot reach this position; here we build an offset mapping by hand
    /// to exercise the empty range through the general code path.
    #[test]
    fn accumulator_fault_above_used_band_is_a_faulty_zero() {
        use super::LiftedRegisterFaultData;
        use crate::fault::{RegisterFault, StuckAt};
        use crate::{Index2, Mapping, Pass};
        use ndarray::array;

        // 2x1 array. The single 1x1 block is placed at row offset 1, leaving
        // array row 0 unused. Clean output is 5 * 3 = 15.
        let weights = array![[5u32]];
        let activations = array![[3u32]];
        let mapping = Mapping::new([Pass::new(0..1, 0..1).with_offset(1, 0)]);

        // Accumulator stuck-at-one on bit 3 at the unused PE above the band.
        // The faulty zero corrupt(0) = 2^3 = 8 flows down into the column sum.
        let targeted = make_targeted_fault(
            Index2 { x: 0, y: 0 },
            PeFaultRegister::Accumulator,
            RegisterFault { stuck_at: StuckAt::One, bit_index: 3 },
        );

        let lifted = mapping.lift_register_fault(&targeted);
        let LiftedRegisterFaultData::Accumulator { parts } = &lifted.data else {
            panic!("expected an accumulator lift");
        };
        assert_eq!(parts.len(), 1);
        assert!(
            parts[0].for_activations.is_empty(),
            "a fault above the band must contribute an empty partial sum"
        );

        let result = lifted.matmul(weights.clone(), activations.clone());
        assert_eq!(result, array![[23u32]]);

        // The lifted result must match the literal array simulation.
        let array = crate::array::SystolicArray::<ArrItem>::new(2, 1).unwrap();
        let expected = literal_matmul(array, targeted, &mapping, &weights, &activations);
        assert_eq!(result, expected);
    }
}
