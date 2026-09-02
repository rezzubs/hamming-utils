use super::{Index2, Pass};

/// Which logic-input regime a [`Pass`] puts a processing element in.
///
/// A pass runs the array at its full physical size on every cycle, so most
/// elements are not necessarily doing that pass's real work. [`matmul`]
/// zero-fills weights and activations outside the pass's own rectangle before
/// running it. This regime says what an element's multiply-add operands
/// actually look like given where it sits relative to that rectangle,
/// independent of whether the element's output ends up contributing to the
/// result.
///
/// [`matmul`]: crate::array::SystolicArray::matmul
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    /// Top row of the pass's rectangle: a real activation and weight, but a
    /// zero incoming partial sum (nothing above it has been loaded this
    /// pass).
    First,
    /// Interior of the pass's rectangle: a real activation, weight, and
    /// incoming partial sum.
    Active,
    /// Above the pass's rectangle, or outside its columns: activation and
    /// weight are both zero, because the pass zero-fills anything it doesn't
    /// use there. Fault-free this element contributes nothing; under a fault
    /// it can still emit non-zero output that propagates through the
    /// elements below it.
    Zero,
    /// Below the pass's rectangle, in one of its columns: activation and
    /// weight are zero, but the incoming partial sum is real. Array output is
    /// read from the physical bottom row, not from the bottom of the
    /// rectangle, so once a column's partial sum is complete it keeps moving
    /// through unloaded rows (computing `0 * 0 + partial_sum`) until it
    /// reaches the bottom.
    Drain,
}

impl Regime {
    /// Which regime `pass` puts the element at `index` in.
    ///
    /// Holds for a both a top-left anchored `pass` as well as one shifted with
    /// [`Pass::with_offset`]: everything here is relative to the pass's own
    /// rectangle ([`Pass::range_y`]/[`Pass::range_x`]), which accounts for
    /// the offset.
    ///
    /// A convenience wrapper exists on [`Pass`] as [`Pass::regime`].
    pub fn classify(pass: &Pass, index: Index2) -> Self {
        let y = usize::from(index.y);
        let x = usize::from(index.x);

        let range_y = pass.range_y();
        let range_x = pass.range_x();

        if !range_x.contains(&x) {
            return Self::Zero;
        }

        if y < range_y.start {
            Self::Zero
        } else if y == range_y.start {
            Self::First
        } else if y < range_y.end {
            Self::Active
        } else {
            Self::Drain
        }
    }
}

/// Which column of the input activation matrix is passing through the element
/// at `index` during `cycle`, or `None` if the element isn't doing real work
/// this cycle (pipeline fill or drain).
///
/// `batch_size` is the activation matrix's column count (`SystolicArray::run`
/// takes an `activations` matrix shaped `(in_features, batch_size)`).
/// Independent of the element's [`Regime`]: every element in the array
/// processes the same sequence of columns, so this only says *which* column is
/// currently there.
pub fn current_column(index: Index2, cycle: usize, batch_size: usize) -> Option<usize> {
    // `run_shifted` shifts activations right and partial sums down by one
    // element per cycle, so a value takes `x + y` cycles to reach this element
    // - that's the earliest cycle it sees real data - and it keeps seeing a new
    // column every cycle after that until all `batch_size` columns have passed
    // through, at which point it's draining.
    let offset = cycle.checked_sub(usize::from(index.x) + usize::from(index.y))?;
    if offset >= batch_size {
        return None;
    }
    // Columns arrive in reverse order: `run_shifted` walks the shifted
    // activation matrix from its last column backwards, feeding one column
    // per cycle. So the earliest real cycle for this element (`offset == 0`)
    // carries the *last* batch column, and the last real cycle carries the
    // first.
    Some(batch_size - 1 - offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::array::{Index2, Pass, PeHook, SystolicArray, cycle_count, shift_activations};
    use crate::test_utilities::ARR_SIZE;
    use ndarray::Array2;
    use proptest::prelude::*;
    use std::ops::{Add, Mul};

    fn index(y: usize, x: usize) -> Index2 {
        Index2 {
            x: x.try_into().expect("index fits in u16"),
            y: y.try_into().expect("index fits in u16"),
        }
    }

    /// Every `(nrows, ncols, batch_size)` triple the exhaustive checks below
    /// enumerate. Kept as a single small range shared by all of them, rather
    /// than three nested loops repeated in each test.
    fn small_configs() -> impl Iterator<Item = (usize, usize, usize)> {
        (1..=6).flat_map(|nrows| {
            (1..=6).flat_map(move |ncols| (1..=6).map(move |batch_size| (nrows, ncols, batch_size)))
        })
    }

    /// Every `(y, x)` element position in an `nrows x ncols` array, as a
    /// single iterator so callers need one loop instead of two.
    fn all_indices(nrows: usize, ncols: usize) -> impl Iterator<Item = (usize, usize)> {
        (0..nrows).flat_map(move |y| (0..ncols).map(move |x| (y, x)))
    }

    /// A matrix where the value at `(y, b)` uniquely encodes `(y, b)`, so a
    /// test can identify which original column reached a given element
    /// without running the array. `+ 1` so `0` unambiguously means "no data
    /// yet" rather than colliding with row/column `0`.
    fn marker_activations(nrows: usize, batch_size: usize) -> Array2<u32> {
        Array2::from_shape_fn((nrows, batch_size), |(y, b)| (y * 100 + b + 1) as u32)
    }

    /// Records, per element, the activation values a real `run_shifted`
    /// execution feeds it, in cycle order. `run_shifted` calls
    /// `multiply_add` exactly once per element per cycle, so the position an
    /// activation lands at in the recorded `Vec` is its cycle number -
    /// nothing here reimplements the array's dataflow, it only observes it.
    struct ActivationRecorder<T> {
        values: Array2<Vec<T>>,
    }

    impl<T> ActivationRecorder<T> {
        fn new(nrows: usize, ncols: usize) -> Self {
            Self {
                values: Array2::from_shape_fn((nrows, ncols), |_| Vec::new()),
            }
        }
    }

    impl<T: Clone> PeHook<T> for ActivationRecorder<T> {
        fn multiply_add(&mut self, index: Index2, activation: T, weight: T, partial_sum: T) -> T
        where
            T: Add<Output = T> + Mul<Output = T>,
        {
            self.values[index].push(activation.clone());
            activation * weight + partial_sum
        }
    }

    /// Cross-checks `current_column` against a real `run_shifted` execution.
    #[test]
    fn activation_feed_matches_current_column() {
        for (nrows, ncols, batch_size) in small_configs() {
            let activations = marker_activations(nrows, batch_size);
            let shifted = shift_activations(&activations);

            let mut array = SystolicArray::<u32>::new(nrows, ncols)
                .expect("small test sizes are always valid")
                .with_hook(ActivationRecorder::new(nrows, ncols));
            array.run_shifted(shifted);

            for (y, x) in all_indices(nrows, ncols) {
                for (cycle, &incoming) in array.hook().values[[y, x]].iter().enumerate() {
                    let expected = current_column(index(y, x), cycle, batch_size)
                        .map(|b| activations[[y, b]])
                        .unwrap_or(0); // The array feeds zeroes after the "real" input.
                    assert_eq!(
                        incoming, expected,
                        "nrows={nrows} ncols={ncols} batch_size={batch_size} y={y} x={x} cycle={cycle}"
                    );
                }
            }
        }
    }

    /// By direct enumeration over every cycle, with no simulator involved: each
    /// element must be doing real work on exactly `batch_size` cycles, since it
    /// sees exactly one activation column per run.
    #[test]
    fn count_invariant_by_enumeration() {
        for (nrows, ncols, batch_size) in small_configs() {
            let cycle_count = cycle_count(nrows, ncols, batch_size);
            for (y, x) in all_indices(nrows, ncols) {
                let used = (0..cycle_count)
                    .filter(|&c| current_column(index(y, x), c, batch_size).is_some())
                    .count();
                assert_eq!(
                    used, batch_size,
                    "PE ({y},{x}) in a {nrows}x{ncols} array, batch_size={batch_size}"
                );
            }
        }
    }

    /// The bottom-right element's last used cycle is exactly the array's last
    /// cycle (`run_shifted`'s `cycle_count - 1`), and no element's window runs
    /// past it. Catches an off-by-one that the count check alone would miss.
    #[test]
    fn tightness() {
        for (nrows, ncols, batch_size) in small_configs() {
            let cycle_count = cycle_count(nrows, ncols, batch_size);

            assert_eq!(
                current_column(index(nrows - 1, ncols - 1), cycle_count - 1, batch_size),
                Some(0)
            );
            assert_eq!(
                current_column(index(nrows - 1, ncols - 1), cycle_count, batch_size),
                None
            );

            for (y, x) in all_indices(nrows, ncols) {
                assert_eq!(
                    current_column(index(y, x), cycle_count, batch_size),
                    None,
                    "PE ({y},{x}) still active past the array's last cycle"
                );
            }
        }
    }

    /// Per element, the used cycles map onto every batch column exactly once.
    ///
    /// Sorting the recorded columns and comparing against `0..batch_size`
    /// catches a duplicate as well as a gap: `current_column` only ever returns
    /// `batch_size` values in total (guaranteed by the `count_invariant` check
    /// above), so a repeated column can only appear by displacing a different
    /// one, which then goes missing from the sort.
    #[test]
    fn bijection_over_batch_columns() {
        for (nrows, ncols, batch_size) in small_configs() {
            let cycle_count = cycle_count(nrows, ncols, batch_size);
            for (y, x) in all_indices(nrows, ncols) {
                let mut columns: Vec<usize> = (0..cycle_count)
                    .filter_map(|c| current_column(index(y, x), c, batch_size))
                    .collect();
                columns.sort_unstable();
                assert_eq!(columns, (0..batch_size).collect::<Vec<_>>(), "PE ({y},{x})");
            }
        }
    }

    #[test]
    fn regime_hand_cases_corner_anchored() {
        // 3x3 array, single pass covering activation rows 0..2 and output
        // columns 0..2.
        let pass = Pass::new(0..2, 0..2);

        assert_eq!(Regime::classify(&pass, index(0, 0)), Regime::First);
        assert_eq!(Regime::classify(&pass, index(0, 1)), Regime::First);
        assert_eq!(Regime::classify(&pass, index(1, 0)), Regime::Active);
        assert_eq!(Regime::classify(&pass, index(1, 1)), Regime::Active);
        assert_eq!(Regime::classify(&pass, index(2, 0)), Regime::Drain);
        assert_eq!(Regime::classify(&pass, index(2, 1)), Regime::Drain);

        // Column 2 is outside the pass's output columns at every row.
        assert_eq!(Regime::classify(&pass, index(0, 2)), Regime::Zero);
        assert_eq!(Regime::classify(&pass, index(1, 2)), Regime::Zero);
        assert_eq!(Regime::classify(&pass, index(2, 2)), Regime::Zero);
    }

    #[test]
    fn regime_hand_cases_offset_pass() {
        // Same pass shape as above, shifted to start at array row/column 1, so
        // row 0 and column 0 are outside the pass's rectangle entirely.
        let pass = Pass::new(0..2, 0..2).with_offset(1, 1);

        assert_eq!(Regime::classify(&pass, index(0, 1)), Regime::Zero);
        assert_eq!(Regime::classify(&pass, index(0, 2)), Regime::Zero);

        assert_eq!(Regime::classify(&pass, index(1, 1)), Regime::First);
        assert_eq!(Regime::classify(&pass, index(1, 2)), Regime::First);
        assert_eq!(Regime::classify(&pass, index(2, 1)), Regime::Active);
        assert_eq!(Regime::classify(&pass, index(2, 2)), Regime::Active);
        assert_eq!(Regime::classify(&pass, index(3, 1)), Regime::Drain);
        assert_eq!(Regime::classify(&pass, index(3, 2)), Regime::Drain);

        // Column 0 is outside range_x at every row, including inside the band.
        assert_eq!(Regime::classify(&pass, index(2, 0)), Regime::Zero);
    }

    #[test]
    fn regime_single_row_band_has_no_active() {
        let pass = Pass::new(0..1, 0..1);
        assert_eq!(Regime::classify(&pass, index(0, 0)), Regime::First);
        assert_eq!(Regime::classify(&pass, index(1, 0)), Regime::Drain);
    }

    /// Pins the branch ordering inside `Regime::classify`: the "outside range_x" check
    /// must run before the row checks, or an idle column below the band would be
    /// misclassified as `Drain` instead of `Zero`.
    #[test]
    fn regime_idle_column_below_band_is_zero_not_drain() {
        let pass = Pass::new(0..2, 0..1);
        assert_eq!(Regime::classify(&pass, index(2, 1)), Regime::Zero);
        assert_eq!(Regime::classify(&pass, index(2, 0)), Regime::Drain);
    }

    fn generate_pass_in_array() -> impl Strategy<Value = (usize, usize, Pass)> {
        (ARR_SIZE, ARR_SIZE).prop_flat_map(|(nrows, ncols)| {
            (0..nrows, 0..ncols).prop_flat_map(move |(row_start, col_start)| {
                (1..=(nrows - row_start), 1..=(ncols - col_start)).prop_map(
                    move |(band_rows, band_cols)| {
                        let pass =
                            Pass::new(0..band_rows, 0..band_cols).with_offset(row_start, col_start);
                        (nrows, ncols, pass)
                    },
                )
            })
        })
    }

    proptest! {
        /// For every column inside the pass's rectangle, the regimes down that
        /// column must appear in exactly the counts the rectangle's geometry
        /// implies: `range_y().start` `Zero`s above it, one `First`, then
        /// `Active` for the rest of the band, then `Drain` down to the bottom.
        /// Every column outside the rectangle is `Zero` at every row. This is a
        /// counting check, not a restatement of `Regime::classify`'s branches, so it
        /// exercises the boundaries independently of how they're implemented.
        #[test]
        fn regime_counts_match_pass_geometry((nrows, ncols, pass) in generate_pass_in_array()) {
            let range_y = pass.range_y();
            let range_x = pass.range_x();

            for x in range_x.clone() {
                let mut zero_count = 0;
                let mut first_count = 0;
                let mut active_count = 0;
                let mut drain_count = 0;

                for y in 0..nrows {
                    match Regime::classify(&pass, index(y, x)) {
                        Regime::Zero => zero_count += 1,
                        Regime::First => first_count += 1,
                        Regime::Active => active_count += 1,
                        Regime::Drain => drain_count += 1,
                    }
                }

                prop_assert_eq!(zero_count, range_y.start);
                prop_assert_eq!(first_count, 1);
                prop_assert_eq!(active_count, range_y.len() - 1);
                prop_assert_eq!(drain_count, nrows - range_y.end);
            }

            for x in 0..ncols {
                if range_x.contains(&x) {
                    continue;
                }
                for y in 0..nrows {
                    prop_assert_eq!(Regime::classify(&pass, index(y, x)), Regime::Zero);
                }
            }
        }
    }
}
