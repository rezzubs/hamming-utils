//! Per-PE profiling of the logic inputs a [`SystolicArray`] sees while running
//! a matmul.
//!
//! [`SystolicArray`]: crate::array::SystolicArray
use ndarray::Array2;

use crate::array::{Index2, Pass, PeHook, Regime, current_column};
use crate::reservoir::Reservoir;

/// The three real-valued operands of one PE's multiply-add:
/// `activation * weight + partial_sum`.
///
/// A named struct rather than a bare `[f32; 3]` so the field order is
/// documented once, in the type itself, instead of by convention at every
/// call site.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LogicInput {
    /// The activation value fed into the PE this cycle.
    pub activation: f32,
    /// The weight value stored in the PE this cycle.
    pub weight: f32,
    /// The partial sum fed into the PE from the element above it.
    pub partial_sum: f32,
}

/// A [`PeHook`] that records, per PE and per input [`Regime`], a bounded
/// uniform sample of the real MAC inputs it sees over one or more [`matmul`]
/// calls.
///
/// "Real" means the observation falls inside the current pass's active cycle
/// window (per [`current_column`]) and outside the [`Regime::Zero`] band, which
/// never carries real inputs. The hook learns the current pass's geometry and
/// batch size from [`PeHook::on_pass_start`], since `multiply_add` alone never
/// sees a pass boundary.
///
/// One RNG stream is shared across every PE and regime, threaded through every
/// `Reservoir::observe` call, rather than seeding one stream per reservoir -
/// matching `Reservoir`'s own design.
///
/// # Panics
///
/// This hook can panic under real usage if it's applied to an array with
/// different dimensions than the hook was initialized with array with different
/// dimensions than the hook was initialized with
///
/// [`matmul`]: crate::array::SystolicArray::matmul
pub struct RecordingHook<R> {
    /// Per-PE sample of [`LogicInput`]s observed while that PE was in the
    /// [`Regime::First`] band.
    first: Array2<Reservoir<LogicInput>>,
    /// Per-PE sample of [`LogicInput`]s observed while that PE was in the
    /// [`Regime::Active`] band.
    active: Array2<Reservoir<LogicInput>>,
    /// Per-PE sample of the incoming partial sum observed while that PE was
    /// in the [`Regime::Drain`] band. Activation and weight are always
    /// `0.0` there, so only the partial sum is worth keeping.
    drain: Array2<Reservoir<f32>>,
    /// Per-PE call count since the last `on_pass_start`. Doubles as the
    /// cycle index: `run_shifted` calls `multiply_add` exactly once per
    /// element per cycle.
    cycles: Array2<usize>,
    /// The pass currently running, set by [`Self::on_pass_start`]. `None`
    /// before the first pass starts.
    pass: Option<Pass>,
    /// The current pass's activation column count, set by
    /// [`Self::on_pass_start`].
    batch_size: usize,
    /// Shared RNG stream, threaded through every reservoir's `observe` call
    /// rather than seeding one stream per reservoir.
    rng: R,
}

impl<R> RecordingHook<R>
where
    R: rand::Rng,
{
    /// Creates a hook for an `nrows x ncols` array, with every PE's per-regime
    /// reservoir bounded to `capacity` samples.
    pub fn new(nrows: usize, ncols: usize, capacity: usize, rng: R) -> Self {
        Self {
            first: Array2::from_shape_fn((nrows, ncols), |_| Reservoir::new(capacity)),
            active: Array2::from_shape_fn((nrows, ncols), |_| Reservoir::new(capacity)),
            drain: Array2::from_shape_fn((nrows, ncols), |_| Reservoir::new(capacity)),
            cycles: Array2::zeros((nrows, ncols)),
            pass: None,
            batch_size: 0,
            rng,
        }
    }

    /// Converts the recorded reservoirs into a plain, serialization-ready
    /// artifact.
    ///
    /// Takes `&self` rather than consuming the hook: a reservoir's retained
    /// samples are cheap to copy out, and the hook is typically still
    /// attached to a [`SystolicArray`] when this is called.
    ///
    /// [`SystolicArray`]: crate::array::SystolicArray
    pub fn to_artifact(&self) -> ProfilingArtifact {
        let (first_triples, first_fill) = reservoirs_to_artifact(&self.first);
        let (active_triples, active_fill) = reservoirs_to_artifact(&self.active);
        let (drain_partial_sums, drain_fill) = reservoirs_to_artifact(&self.drain);
        ProfilingArtifact {
            first_triples,
            first_fill,
            active_triples,
            active_fill,
            drain_partial_sums,
            drain_fill,
        }
    }
}

impl<R: rand::Rng> PeHook<f32> for RecordingHook<R> {
    fn on_pass_start(&mut self, pass: &Pass, batch_size: usize) {
        self.pass = Some(pass.clone());
        self.batch_size = batch_size;
        self.cycles.fill(0);
    }

    fn multiply_add(
        &mut self,
        index: Index2,
        activation: f32,
        weight: f32,
        partial_sum: f32,
    ) -> f32 {
        let result = activation * weight + partial_sum;

        let cycle = self.cycles[index];
        self.cycles[index] += 1;

        let pass = self.pass.as_ref().expect(
            "on_pass_start is always called before multiply_add, from within SystolicArray::matmul's pass loop",
        );

        if current_column(index, cycle, self.batch_size).is_some() {
            match pass.regime(index) {
                // Zero is left deliberately empty. All logic inputs are
                // guaranteed to be zero so there's nothing to record.
                Regime::Zero => {}
                Regime::First => {
                    self.first[index].observe(
                        LogicInput {
                            activation,
                            weight,
                            partial_sum,
                        },
                        &mut self.rng,
                    );
                }
                Regime::Active => {
                    self.active[index].observe(
                        LogicInput {
                            activation,
                            weight,
                            partial_sum,
                        },
                        &mut self.rng,
                    );
                }
                Regime::Drain => {
                    self.drain[index].observe(partial_sum, &mut self.rng);
                }
            }
        }

        result
    }
}

/// The recorded per-PE distribution of MAC inputs, split by [`Regime`],
/// ready to serialize.
///
/// [`Regime::Zero`] stores nothing: fault-free it is always exactly
/// `LogicInput::default()`, so there is nothing to sample. Each `*_fill`
/// entry is `min(capacity, observations)` for that PE - the number of real
/// samples in the matching `*_triples`/`*_partial_sums` entry.
#[derive(Debug, Clone)]
pub struct ProfilingArtifact {
    /// Per-PE sample of [`LogicInput`]s observed in the [`Regime::First`]
    /// band.
    pub first_triples: Array2<Vec<LogicInput>>,
    /// Per-PE number of real samples in `first_triples` (`min(capacity,
    /// observations)`).
    pub first_fill: Array2<usize>,
    /// Per-PE sample of [`LogicInput`]s observed in the [`Regime::Active`]
    /// band.
    pub active_triples: Array2<Vec<LogicInput>>,
    /// Per-PE number of real samples in `active_triples`.
    pub active_fill: Array2<usize>,
    /// Per-PE sample of incoming partial sums observed in the
    /// [`Regime::Drain`] band.
    pub drain_partial_sums: Array2<Vec<f32>>,
    /// Per-PE number of real samples in `drain_partial_sums`.
    pub drain_fill: Array2<usize>,
}

fn reservoirs_to_artifact<T: Clone + Default>(
    grid: &Array2<Reservoir<T>>,
) -> (Array2<Vec<T>>, Array2<usize>) {
    let samples = grid.map(|reservoir| reservoir.samples().to_vec());
    let fill = grid.map(Reservoir::sample_count);
    (samples, fill)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::{Mapping, SystolicArray};
    use crate::test_utilities::index;
    use ndarray::array;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Kept smaller than `ARR_SIZE` (which goes up to 32) so property tests
    /// stay fast - profiling every PE with a per-cycle reservoir observation
    /// isn't as cheap as a plain matmul, and correctness here doesn't depend on
    /// array size.
    const SMALL_SIZE: std::ops::RangeInclusive<usize> = 1..=6;

    fn generate_f32_array() -> impl Strategy<Value = SystolicArray<f32>> {
        (SMALL_SIZE, SMALL_SIZE)
            .prop_map(|(height, width)| SystolicArray::<f32>::new(height, width).unwrap())
    }

    fn generate_f32_weights_and_activations() -> impl Strategy<Value = (Array2<f32>, Array2<f32>)> {
        (SMALL_SIZE, SMALL_SIZE, SMALL_SIZE).prop_flat_map(
            |(in_features, out_features, batch_size)| {
                let value = -10.0f32..10.0;
                let weights = proptest::collection::vec(value.clone(), in_features * out_features)
                    .prop_map(move |v| {
                        Array2::from_shape_vec((out_features, in_features), v).unwrap()
                    });
                let activations = proptest::collection::vec(value, in_features * batch_size)
                    .prop_map(move |v| {
                        Array2::from_shape_vec((in_features, batch_size), v).unwrap()
                    });
                (weights, activations)
            },
        )
    }

    /// Like [`generate_f32_weights_and_activations`], but values are never
    /// exactly zero - needed by the all-zero-census test, which checks that
    /// no all-zero `LogicInput` leaks into `active_triples`.
    fn generate_nonzero_f32_weights_and_activations()
    -> impl Strategy<Value = (Array2<f32>, Array2<f32>)> {
        (SMALL_SIZE, SMALL_SIZE, SMALL_SIZE).prop_flat_map(
            |(in_features, out_features, batch_size)| {
                let value = 1.0f32..10.0;
                let weights = proptest::collection::vec(value.clone(), in_features * out_features)
                    .prop_map(move |v| {
                        Array2::from_shape_vec((out_features, in_features), v).unwrap()
                    });
                let activations = proptest::collection::vec(value, in_features * batch_size)
                    .prop_map(move |v| {
                        Array2::from_shape_vec((in_features, batch_size), v).unwrap()
                    });
                (weights, activations)
            },
        )
    }

    proptest! {
        /// Recording must never change the result of a matmul: the hook's
        /// `multiply_add` computes the exact same `a * w + partial_sum` as the
        /// trait default, in the same order, so the two paths must agree
        /// bit-for-bit.
        #[test]
        fn passthrough_matches_plain_matmul(
            mut arr in generate_f32_array(),
            (weights, activations) in generate_f32_weights_and_activations(),
        ) {
            let mapping = arr.auto_mapping_for(&weights);
            let expected = arr.matmul(&mapping, &weights, &activations);

            let hook = RecordingHook::new(arr.nrows(), arr.ncols(), 8, StdRng::seed_from_u64(0));
            let mut recording_arr = arr.with_hook(hook);
            let actual = recording_arr.matmul(&mapping, &weights, &activations);

            prop_assert_eq!(actual, expected);
        }

        /// No all-zero [`LogicInput`] should ever land in `active_triples`
        /// when the real weights/activations never contain a zero - such a
        /// value could only come from a zero-padding cycle being
        /// misclassified as real (a cycle-window *offset* error), since a
        /// genuine ACTIVE observation always has a real, nonzero activation.
        #[test]
        fn active_reservoir_has_no_all_zero_triples(
            arr in generate_f32_array(),
            (weights, activations) in generate_nonzero_f32_weights_and_activations(),
        ) {
            let mapping = arr.auto_mapping_for(&weights);
            let hook = RecordingHook::new(arr.nrows(), arr.ncols(), 8, StdRng::seed_from_u64(0));
            let mut arr = arr.with_hook(hook);
            arr.matmul(&mapping, &weights, &activations);

            let artifact = arr.hook().to_artifact();
            for triples in &artifact.active_triples {
                for triple in triples {
                    prop_assert_ne!(*triple, LogicInput::default());
                }
            }
        }

        /// Every PE the mapping puts in a non-`Zero` regime for some pass
        /// must end up with a nonzero fill somewhere; every PE that is
        /// `Zero` for every pass touching it must have zero fill everywhere.
        /// Checkable by direct enumeration over the mapping's own regimes.
        #[test]
        fn coverage_matches_mapping_regimes(
            arr in generate_f32_array(),
            (weights, activations) in generate_f32_weights_and_activations(),
        ) {
            let mapping = arr.auto_mapping_for(&weights);
            let hook = RecordingHook::new(arr.nrows(), arr.ncols(), 4, StdRng::seed_from_u64(0));
            let mut arr = arr.with_hook(hook);
            arr.matmul(&mapping, &weights, &activations);
            let artifact = arr.hook().to_artifact();

            for y in 0..artifact.first_fill.nrows() {
                for x in 0..artifact.first_fill.ncols() {
                    let index = index(y, x);
                    let total_fill = artifact.first_fill[[y, x]]
                        + artifact.active_fill[[y, x]]
                        + artifact.drain_fill[[y, x]];

                    let touched_non_zero =
                        (&mapping).into_iter().any(|pass| pass.regime(index) != Regime::Zero);

                    if touched_non_zero {
                        prop_assert!(total_fill > 0, "PE ({}, {}) should have been recorded", y, x);
                    } else {
                        prop_assert_eq!(total_fill, 0, "PE ({}, {}) should be untouched", y, x);
                    }
                }
            }
        }
    }

    /// Hand-built: 3x3 array, single pass covering activation rows 0..2 and
    /// output columns 0..2, so row 0 is First, row 1 Active, row 2 Drain, and
    /// column 2 is Zero throughout. Every First/Active/Drain PE must record
    /// exactly `batch_size` observations - checkable without a netlist, and
    /// catches cycle-window *width* errors immediately.
    #[test]
    fn count_matches_batch_size_per_regime() {
        let batch_size = 4;
        let pass = Pass::new(0..2, 0..2);
        let mapping = Mapping::new([pass.clone()]);

        let weights = Array2::<f32>::zeros((2, 2));
        let activations = Array2::<f32>::zeros((2, batch_size));

        let hook = RecordingHook::new(3, 3, 8, StdRng::seed_from_u64(0));
        let mut arr = SystolicArray::<f32>::new(3, 3).unwrap().with_hook(hook);
        arr.matmul(&mapping, &weights, &activations);
        let artifact = arr.hook().to_artifact();

        for y in 0..3 {
            for x in 0..3 {
                let index = Index2 {
                    x: x as u16,
                    y: y as u16,
                };
                match pass.regime(index) {
                    Regime::First => assert_eq!(artifact.first_fill[[y, x]], batch_size),
                    Regime::Active => assert_eq!(artifact.active_fill[[y, x]], batch_size),
                    Regime::Drain => assert_eq!(artifact.drain_fill[[y, x]], batch_size),
                    Regime::Zero => {
                        assert_eq!(artifact.first_fill[[y, x]], 0);
                        assert_eq!(artifact.active_fill[[y, x]], 0);
                        assert_eq!(artifact.drain_fill[[y, x]], 0);
                    }
                }
            }
        }
    }

    /// Hand-built, with an offset pass so row/column 0 fall outside the pass's
    /// rectangle entirely to confirm coverage holds for offset passes, not just
    /// corner-anchored ones.
    #[test]
    fn coverage_matches_mapping_regimes_with_offset_pass() {
        let pass = Pass::new(0..2, 0..2).with_offset(1, 1);
        let mapping = Mapping::new([pass.clone()]);

        let weights = Array2::<f32>::from_elem((2, 2), 1.0);
        let activations = Array2::<f32>::from_elem((2, 3), 1.0);

        let hook = RecordingHook::new(4, 4, 8, StdRng::seed_from_u64(0));
        let mut arr = SystolicArray::<f32>::new(4, 4).unwrap().with_hook(hook);
        arr.matmul(&mapping, &weights, &activations);
        let artifact = arr.hook().to_artifact();

        for y in 0..4 {
            for x in 0..4 {
                let index = Index2 {
                    x: x as u16,
                    y: y as u16,
                };
                let total_fill = artifact.first_fill[[y, x]]
                    + artifact.active_fill[[y, x]]
                    + artifact.drain_fill[[y, x]];

                if pass.regime(index) == Regime::Zero {
                    assert_eq!(total_fill, 0, "PE ({y},{x}) should be untouched");
                } else {
                    assert!(total_fill > 0, "PE ({y},{x}) should have been recorded");
                }
            }
        }
    }

    /// Validates the partial-sum chain endpoint, the regime classification, and the
    /// column-to-output-row mapping against a computation that shares no
    /// reasoning with the profiler or the array simulator: a plain
    /// `ndarray` `.dot()`.
    ///
    /// Same 3x2 geometry as `count_matches_batch_size_per_regime` (row
    /// 0 First, row 1 Active, row 2 Drain), with a reservoir capacity
    /// `>= batch_size` so - per `Reservoir`'s own proven property - no
    /// eviction ever happens and `samples()` holds every real observation
    /// in arrival order. `current_column`'s own contract says arrival order
    /// is *reverse* batch-column order (the earliest cycle carries the
    /// last column), so `drain_partial_sums[[drain_row, x]][i]` must equal
    /// `expected[[output_row, batch_size - 1 - i]]`.
    #[test]
    fn drain_reservoir_matches_independent_matmul() {
        let batch_size = 3;
        let capacity = 8;

        let weights = array![[1.0f32, 2.0], [3.0, 4.0]];
        let activations = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let pass = Pass::new(0..2, 0..2);
        let mapping = Mapping::new([pass.clone()]);

        let hook = RecordingHook::new(3, 2, capacity, StdRng::seed_from_u64(0));
        let mut arr = SystolicArray::<f32>::new(3, 2).unwrap().with_hook(hook);
        arr.matmul(&mapping, &weights, &activations);
        let artifact = arr.hook().to_artifact();

        let expected = weights.dot(&activations);
        let drain_row = pass.range_y().end;

        for x in 0..2 {
            let output_row = pass
                .output_row_from_array_col(x)
                .expect("x is inside range_x");
            let samples = &artifact.drain_partial_sums[[drain_row, x]];
            assert_eq!(samples.len(), batch_size);
            for (i, &partial_sum) in samples.iter().enumerate() {
                let batch_column = batch_size - 1 - i;
                assert_eq!(partial_sum, expected[[output_row, batch_column]]);
            }
        }
    }
}
