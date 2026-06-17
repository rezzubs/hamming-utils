#[cfg(test)]
mod tests;

use std::{
    collections::{HashMap, HashSet},
    ops::Range,
};

use ndarray::prelude::*;

/// A single pass through a systolic array. Describes a full or partial matrix multiplication.
///
/// The components describe which input (activation) rows will be passed through
/// the systolic array and which outputs will be produced. Activation rows are
/// mapped to rows in the array and output rows are mapped to columns.
///
/// The ranges will be placed at the top-left location in the array and
/// optionally shifted by `array_row_start` and `array_col_start`. For example,
/// the activation range 5..9 will map the activation rows 5, 6, 7, 8 to the
/// systolic array rows `0 + array_row_start` through `3 + array_row_start`.
///
/// These ranges are expected to be consistent across passes. For example if a
/// range maps the activation index 5 to the array index 0, all further uses of
/// activation index 5 need to map the same way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pass {
    /// An offset for the mapping of `activation_rows`. The `activation_rows`
    /// will be mapped to array rows starting from `0 + array_row_start`.
    pub array_row_start: usize,
    /// The activation matrix rows that will be mapped to the systolic array
    /// rows during this pass.
    pub activation_rows: Range<usize>,
    /// An offset for the mapping of `output_rows`. The `output_rows` will be
    /// mapped to array rows starting from `0 + array_col_start`.
    pub array_col_start: usize,
    /// The output matrix rows that will be mapped to the systolic array columns
    /// during this pass.
    pub output_rows: Range<usize>,
}

/// Index into a range.
///
/// # Panics
///
/// Panics if `range.end <= range.start`.
fn range_get(range: &Range<usize>, index: usize) -> Option<usize> {
    assert!(range.end > range.start);

    if index >= range.len() {
        None
    } else {
        Some(index + range.start)
    }
}

impl Pass {
    /// Create a new pass.
    pub fn new(activation_rows: Range<usize>, output_rows: Range<usize>) -> Self {
        Self {
            array_row_start: 0,
            activation_rows,
            array_col_start: 0,
            output_rows,
        }
    }

    /// Add an offset to the pass mapping.
    pub fn with_offset(mut self, y: usize, x: usize) -> Self {
        self.array_row_start = y;
        self.array_col_start = x;
        self
    }

    /// Returns the number of rows the pass maps to.
    pub fn nrows(&self) -> usize {
        self.array_row_start + self.activation_rows.len()
    }

    /// Returns the number of columns the pass maps to.
    pub fn ncols(&self) -> usize {
        self.array_col_start + self.output_rows.len()
    }

    /// Returns the range of mapped y indices of the systolic array.
    pub fn range_y(&self) -> Range<usize> {
        self.array_row_start..(self.array_row_start + self.activation_rows.len())
    }

    /// Returns the range of mapped x indices of the systolic array.
    pub fn range_x(&self) -> Range<usize> {
        self.array_col_start..(self.array_col_start + self.output_rows.len())
    }

    /// Checks which activation row (if any) is mapped to an input row of the array.
    pub fn activation_row_from_array_row(&self, input_row: usize) -> Option<usize> {
        let shifted_index = input_row.checked_sub(self.array_row_start)?;
        range_get(&self.activation_rows, shifted_index)
    }

    pub fn output_row_from_array_col(&self, input_col: usize) -> Option<usize> {
        let shifted_index = input_col.checked_sub(self.array_col_start)?;
        range_get(&self.output_rows, shifted_index)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InvalidMappingError {
    #[error("connection {activation_row}->{output_row} appears in multiple passes")]
    DuplicateConnection {
        activation_row: usize,
        output_row: usize,
    },
    #[error("the mapping is missing connections: {0:?}")]
    MissingConnections(HashSet<Connection>),
    #[error(
        "the activation row {activation_row} is mapped to different array rows: {array_row1} (pass {array_row1_pass}) and {array_row2} (pass {array_row2_pass})"
    )]
    InconsistentActivation {
        activation_row: usize,
        array_row1: usize,
        array_row1_pass: usize,
        array_row2: usize,
        array_row2_pass: usize,
    },
    #[error(
        "the output row {output_row} is mapped to different array columns: {array_col1} (pass {array_col1_pass}) and {array_col2} (pass {array_col2_pass})"
    )]
    InconsistentOutput {
        output_row: usize,
        array_col1: usize,
        array_col1_pass: usize,
        array_col2: usize,
        array_col2_pass: usize,
    },
}

/// Describes how a matrix multiplication is mapped to a systolic array.
///
/// The mapping is defined as a series of passes through the array. Each
/// [`Pass`] describes a full or partial matrix multiplication. A single pass
/// describes which activation rows are passed through the array and which
/// output rows are produced.
///
/// If the same output row appears in multiple passes the results are accumulated.
/// This can be used to split operations, which might otherwise not fit, across
/// multiple passes.
///
/// # Constructing a valid mapping
///
/// - A valid mapping must connect all input rows to all output rows but it
///   doesn't matter when and in which order each connection happens (assuming no
///   faults). An input row is considered connected to an output row if they
///   appear together in a pass.
///
/// - The combination of an input row and output row must appear in exactly one
///   pass. For example, if a pass includes input row `0` and output row `5` then
///   no other passes can contain both at the same time, but it's fine for a pass
///   to include them individually.
///
/// - If one pass maps the activation row `a` to the array row `b`, then all
///   following appearances of `a` need to map to `b` as well. The same is true
///   for mapping output rows to array columns.
///
/// - The mapping needs to contain at least one pass that connects at least one
///   input row to an output row.
///
/// The validity of a mapping can be verified by the [`Mapping::validate`]
/// method.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mapping {
    passes: Vec<Pass>,
}

impl Mapping {
    /// Create a new [`Mapping`] from an iterator of [`Pass`]es.
    ///
    /// Returns None if `passes` doesn't produce any elements.
    pub fn new_checked(passes: impl IntoIterator<Item = Pass>) -> Option<Self> {
        let passes = passes.into_iter().collect::<Vec<_>>();
        if passes.is_empty() {
            return None;
        }

        Some(Self { passes })
    }

    /// Create a new [`Mapping`] from an iterator of [`Pass`]es.
    ///
    /// # Panics
    ///
    /// Panics if `passes` doesn't produce any elements.
    pub fn new(passes: impl IntoIterator<Item = Pass>) -> Self {
        Self::new_checked(passes).expect("Cannot create a mapping with no passes")
    }

    /// Create a mapping for the given matmul input-row and output-row counts.
    ///
    /// The weight matrix will be partitioned into blocks. The algorithm tries
    /// to maximize the size of the blocks. As many systolic-array-shaped blocks will be used
    ///
    /// For example the following 3x4 matrix
    ///
    /// ```text
    /// |  1  2  3  4 |
    /// |  5  6  7  8 |
    /// |  9 10 11 12 |
    /// ```
    ///
    /// will be mapped to a 3x2 array like so:
    ///
    /// ```text
    /// |-----|----|
    /// | 1 5 |  9 |
    /// | 2 6 | 10 |
    /// | 3 7 | 11 |
    /// |-----|----|
    /// | 4 8 | 12 |
    /// |-----|----|
    /// ```
    ///
    /// Remember that the weight matrix sits transposed inside the array.
    ///
    /// These blocks will become passes in a row-major order. Left-to-right in
    /// each row, then shift down to the next row.
    ///
    /// See also [`Self::auto_for`].
    pub fn auto(
        in_features: usize,
        out_features: usize,
        array_nrows: usize,
        array_ncols: usize,
    ) -> Self {
        let columns_in_full_block = array_ncols;
        let full_blocks_horizontal = out_features / columns_in_full_block;
        let columns_in_partial_block = out_features % columns_in_full_block;

        let rows_in_full_block = array_nrows;
        let full_blocks_vertical = in_features / rows_in_full_block;
        let rows_in_partial_block = in_features % rows_in_full_block;

        let mut ranges_x = Vec::new();
        for block_x in 0..full_blocks_horizontal {
            let start = block_x * columns_in_full_block;
            let end = start + columns_in_full_block;
            ranges_x.push(start..end)
        }
        if columns_in_partial_block > 0 {
            let start = full_blocks_horizontal * columns_in_full_block;
            let end = start + columns_in_partial_block;
            ranges_x.push(start..end);
        }

        let mut ranges_y = Vec::new();
        for block_y in 0..full_blocks_vertical {
            let start = block_y * rows_in_full_block;
            let end = start + rows_in_full_block;
            ranges_y.push(start..end)
        }
        if rows_in_partial_block > 0 {
            let start = full_blocks_vertical * rows_in_full_block;
            let end = start + rows_in_partial_block;
            ranges_y.push(start..end);
        }

        let mut passes = Vec::new();
        for activation_rows in ranges_y {
            for output_rows in &ranges_x {
                passes.push(Pass::new(activation_rows.clone(), output_rows.clone()))
            }
        }

        Mapping::new(passes)
    }

    /// Create a mapping for the given weights.
    ///
    /// `weights` are assumed to be a matrix with shape `(out_features, in_features)`.
    ///
    /// See also [`Self::auto`].
    pub fn auto_for<T>(weights: &ArrayRef2<T>, array_nrows: usize, array_ncols: usize) -> Self {
        let out_features = weights.nrows();
        let in_features = weights.ncols();
        Self::auto(in_features, out_features, array_nrows, array_ncols)
    }

    /// The minimum array size required by this mapping.
    pub fn array_size_min(&self) -> (usize, usize) {
        (self.array_nrows_min(), self.array_ncols_min())
    }

    /// The minimum number of array rows required by this mapping.
    pub fn array_nrows_min(&self) -> usize {
        self.passes
            .iter()
            .map(|pass| pass.nrows())
            .max()
            .expect("Mapping without passes")
    }

    /// The minimum number of array columns required by this mapping.
    pub fn array_ncols_min(&self) -> usize {
        self.passes
            .iter()
            .map(|pass| pass.ncols())
            .max()
            .expect("Mapping without passes")
    }

    /// Check "data index" -> "array index" consistency across passes.
    fn validate_consistency(&self) -> Result<(), InvalidMappingError> {
        let mut column_mappings = HashMap::<usize, (usize, usize)>::new();
        let mut row_mappings = HashMap::<usize, (usize, usize)>::new();

        use std::collections::hash_map::Entry;
        for (pass_index, pass) in self.passes.iter().enumerate() {
            for (array_row_raw, activation_row) in pass.activation_rows.clone().enumerate() {
                let array_row = pass.array_row_start + array_row_raw;

                match row_mappings.entry(activation_row) {
                    Entry::Occupied(entry) => {
                        let (previous_pass_index, previous) = *entry.get();

                        if previous != array_row {
                            return Err(InvalidMappingError::InconsistentActivation {
                                activation_row,
                                array_row1: previous,
                                array_row1_pass: previous_pass_index,
                                array_row2: array_row,
                                array_row2_pass: pass_index,
                            });
                        }
                    }
                    Entry::Vacant(entry) => {
                        entry.insert((pass_index, array_row));
                    }
                }
            }

            for (array_col_raw, output_row) in pass.output_rows.clone().enumerate() {
                let array_col = pass.array_col_start + array_col_raw;

                match column_mappings.entry(output_row) {
                    Entry::Occupied(entry) => {
                        let (previous_pass_index, previous) = *entry.get();

                        if previous != array_col {
                            return Err(InvalidMappingError::InconsistentOutput {
                                output_row,
                                array_col1: previous,
                                array_col1_pass: previous_pass_index,
                                array_col2: array_col,
                                array_col2_pass: pass_index,
                            });
                        }
                    }
                    Entry::Vacant(entry) => {
                        entry.insert((pass_index, array_col));
                    }
                }
            }
        }

        Ok(())
    }

    /// Validates the mapping by ensuring all input-row/output-row connections are present
    /// exactly once across all passes.
    pub fn validate(&self) -> Result<(), InvalidMappingError> {
        let activation_index_max = self
            .passes
            .iter()
            .map(|pass| pass.activation_rows.end)
            .max()
            .expect("Mapping without passes")
            - 1;
        let output_index_max = self
            .passes
            .iter()
            .map(|pass| pass.output_rows.end)
            .max()
            .expect("Mapping without passes")
            - 1;

        let mut missing_connections = HashSet::<Connection>::with_capacity(
            (activation_index_max + 1) * (output_index_max + 1),
        );

        for expected_input_row in 0..=activation_index_max {
            for expected_output_row in 0..=output_index_max {
                let fresh = missing_connections.insert(Connection {
                    activation_feature: expected_input_row,
                    output_feature: expected_output_row,
                });
                debug_assert!(
                    fresh,
                    "connection already set: {}->{}",
                    expected_input_row, expected_output_row
                );
            }
        }

        for pass in self.passes.iter() {
            for activation_row in pass.activation_rows.clone() {
                for output_row in pass.output_rows.clone() {
                    let present =
                        missing_connections.remove(&Connection::new(activation_row, output_row));

                    if !present {
                        return Err(InvalidMappingError::DuplicateConnection {
                            activation_row,
                            output_row,
                        });
                    }
                }
            }
        }

        if !missing_connections.is_empty() {
            return Err(InvalidMappingError::MissingConnections(missing_connections));
        }

        self.validate_consistency()?;

        Ok(())
    }
}

impl IntoIterator for Mapping {
    type Item = Pass;

    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.passes.into_iter()
    }
}

impl<'a> IntoIterator for &'a Mapping {
    type Item = &'a Pass;

    type IntoIter = std::slice::Iter<'a, Pass>;

    fn into_iter(self) -> Self::IntoIter {
        self.passes.iter()
    }
}

/// A connection between a matmul input (activation) row and a matmul output row within a [`Pass`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Connection {
    pub activation_feature: usize,
    pub output_feature: usize,
}

impl Connection {
    pub fn new(input_row: usize, output_row: usize) -> Self {
        Self {
            activation_feature: input_row,
            output_feature: output_row,
        }
    }
}
