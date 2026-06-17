mod index;
mod mapping;
mod register;
#[cfg(test)]
pub(crate) mod tests;

pub use index::{Index, Index2};
pub use mapping::{Connection, InvalidMappingError, Mapping, Pass};
use ndarray::prelude::*;
use register::Register;
use std::{
    fmt::Debug,
    ops::{AddAssign, Mul},
};

#[derive(Debug, PartialEq, Eq, Clone, Copy, Default)]
struct ProcessingElement<T> {
    weight: Register<T>,
    activation: Register<T>,
    partial_sum: Register<T>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CreationError {
    #[error("array must have at least one row")]
    NoRows,
    #[error("array must have at least one column")]
    NoCols,
    #[error("cannot create u16 indexes for {0} rows")]
    TooManyRows(usize),
    #[error("cannot create u16 indexes for {0} columns")]
    TooManyCols(usize),
}

/// A simulator for a systolic array.
///
/// This type exists to validate [`crate::mapping::Mapping`]s, .
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystolicArray<T> {
    /// The processing elements that make up the array.
    elements: Array2<ProcessingElement<T>>,
}

impl<T> SystolicArray<T>
where
    T: Default + Clone,
{
    /// Create a new array.
    pub fn new(nrows: usize, ncols: usize) -> Result<Self, CreationError> {
        if nrows == 0 {
            return Err(CreationError::NoRows);
        }
        if ncols == 0 {
            return Err(CreationError::NoCols);
        }
        let max_index = Index::MAX as usize + 1;
        if nrows > max_index {
            return Err(CreationError::TooManyRows(nrows));
        }
        if ncols > max_index {
            return Err(CreationError::TooManyCols(ncols));
        }

        Ok(Self {
            elements: Array2::from_elem((nrows, ncols), ProcessingElement::default()),
        })
    }

    /// Create a new array with the given weights.
    ///
    /// `weights` are assumed to be a matrix with shape `(out_features, in_features)`.
    ///
    /// # Panics if the size is great enough so it cannot be indexed by
    /// ([`Index2`], [`Index2`]).
    pub fn from_weights(weights: &ArrayRef2<T>) -> Self {
        assert!(weights.nrows() < Index::MAX as usize);
        assert!(weights.ncols() < Index::MAX as usize);

        // The weights need to be transposed if we want to feed activations
        // without modifying their shape.
        let mut array = Self::new(weights.ncols(), weights.nrows()).unwrap();
        array.set_weights(weights);
        array
    }

    /// Returns the number of rows in the array.
    pub fn nrows(&self) -> usize {
        self.elements.nrows()
    }

    /// Returns the number of columns in the array.
    pub fn ncols(&self) -> usize {
        self.elements.ncols()
    }

    /// Sets the weights of the array. Expects a transposed weight matrix.
    ///
    /// `weights_raw` are assumed to be a matrix with shape `(in_features, out_features)`.
    ///
    /// If there is a fault in one of the weight registers, all weight values
    /// below that register will be corrupted. This is because a real systolic
    /// array would read in weights from the top, similar to how the activation
    /// values are fed in from the left during execution.
    ///
    /// See also: [`Self::set_weights`].
    pub fn set_weights_raw(&mut self, weights_raw: &ArrayRef2<T>) {
        assert_eq!(self.elements.shape(), weights_raw.shape());

        for (index, weight) in weights_raw.indexed_iter() {
            self.elements[index].weight.write(weight.clone());
        }
    }

    /// Set the weights of the array with a weight matrix.
    ///
    /// `weights` are assumed to be a matrix with shape `(out_features, in_features)`.
    ///
    /// See also: [`Self::set_weights_raw`].
    pub fn set_weights(&mut self, weights: &ArrayRef2<T>) {
        self.set_weights_raw(&weights.t());
    }

    /// Run a matrix multiplication with a matrix which has already been
    /// row-shifted.
    ///
    /// For example the activation matrix
    ///
    /// ```text
    /// a b
    /// c d
    /// ```
    ///
    /// would need to be written as
    ///
    /// ```text
    /// 0 a b
    /// c d 0
    /// ```
    ///
    /// This operation is not really pure as the processing element activation
    /// registers can be left in a different state than the initial conditions.
    /// The output, excluding the padding, is guaranteed to be correct no matter
    /// the current register state. If activation padding contains values other
    /// than zero then the output padding can contain non-zero values on
    /// subsequent runs. Padding with zeros ensures zero-padded outputs over
    /// many runs.
    pub fn run_shifted(&mut self, row_shifted_activations: Array2<T>) -> Array2<T>
    where
        T: num_traits::Zero + AddAssign + Mul<Output = T>,
    {
        assert_eq!(self.nrows(), row_shifted_activations.nrows());
        assert!(row_shifted_activations.nrows() <= row_shifted_activations.ncols());

        // The number of steps the top activation has to take to reach the
        // bottom right PE.
        let longest_path_through_array = self.ncols() + self.nrows() - 1;
        let batches_count = row_shifted_activations.ncols() - row_shifted_activations.nrows() + 1;
        // how many iterations we have to evaluate the array to produce the full
        // output. 1 extra step for every batch after the first
        let cycle_count = longest_path_through_array + batches_count - 1;

        // The cycle index at which the first output element appears.
        let output_start_cycle = self.nrows() - 1;

        let mut output_buffer =
            Array2::<T>::zeros((self.ncols() + batches_count - 1, self.ncols()));

        // An array of zeroes is fed into the array in place of activations when
        // activation values have been exhausted but the array has not yet
        // finished cycling.
        let zero_activations = Array1::<T>::zeros(self.nrows());

        for cycle in 0..cycle_count {
            // we start with the last column and work backwards switching to
            // zeroes once exhausted.
            let target_activation_column = (row_shifted_activations.ncols() - 1)
                .checked_sub(cycle)
                .map(|index| row_shifted_activations.column(index))
                .unwrap_or_else(|| zero_activations.view());

            // Need to iterate backwards in both axes to not overwrite the
            // partial sums from the previous cycle.
            for y in (0..self.nrows()).rev() {
                for x in (0..self.ncols()).rev() {
                    let current_element_index = [y, x];

                    // Shift activations right
                    let left_element_activation_value = if x == 0 {
                        target_activation_column[y].clone()
                    } else {
                        let left_element_index = [y, x - 1];
                        self.elements[left_element_index].activation.read()
                    };
                    self.elements[current_element_index]
                        .activation
                        .write(left_element_activation_value);

                    // Compute a new partial sum for this element
                    let (activation, weight) = {
                        let element = &self.elements[current_element_index];
                        (element.activation.read(), element.weight.read())
                    };
                    let partial_sum_above = if y == 0 {
                        T::zero()
                    } else {
                        let index_above = [y - 1, x];
                        self.elements[index_above].partial_sum.read()
                    };

                    self.elements[[y, x]]
                        .partial_sum
                        .write(activation * weight + partial_sum_above);
                }
            }

            if cycle < output_start_cycle {
                continue;
            }

            let output_y_max = output_buffer.nrows() - 1;
            let offset = cycle - output_start_cycle;
            let output_y = output_y_max - offset;

            // Copy the bottom row of partial sums into the output buffer.
            output_buffer.row_mut(output_y).assign(
                &self
                    .elements
                    .row(self.nrows() - 1)
                    .map(|e| e.partial_sum.read()),
            );
        }

        output_buffer
    }

    /// Run the array with a matrix of activations.
    ///
    /// `activations` are assumed to be a matrix with shape `(in_features, batch_size)`.
    ///
    /// See also [`Self::run_shifted`].
    pub fn run(&mut self, activations: &ArrayRef2<T>) -> Array2<T>
    where
        T: num_traits::Zero + AddAssign + Mul<Output = T>,
    {
        unshift_output(&self.run_shifted(shift_activations(activations)))
    }

    /// Generate a mapping for the given matmul input-row and output-row counts.
    ///
    /// See [`Mapping::auto`] for details of the algorithm.
    ///
    /// See also [`Self::auto_mapping_for`].
    pub fn auto_mapping(&self, in_features: usize, out_features: usize) -> Mapping {
        Mapping::auto(in_features, out_features, self.nrows(), self.ncols())
    }

    /// Generate a mapping for the given weights.
    ///
    /// `weights` are assumed to be an array with shape `(out_features, in_features)`.
    ///
    /// See [`Mapping::auto`] for details of the algorithm.
    ///
    /// See also [`Self::auto_mapping`].
    pub fn auto_mapping_for(&self, weights: &ArrayRef2<T>) -> Mapping {
        Mapping::auto_for(weights, self.nrows(), self.ncols())
    }

    /// Check if the given mapping is usable on this array.
    pub fn supports_mapping(&self, mapping: &Mapping) -> bool {
        let (nrows, ncols) = mapping.array_size_min();

        !(nrows > self.nrows() || ncols > self.ncols())
    }

    /// Perform a matrix multiplication using the given mapping and
    /// activations/weights.
    ///
    /// # Panics
    ///
    /// Panics if the mapping isn't supported by this array. It is possible to
    /// validate the mapping beforehand using [`Self::supports_mapping`]. This
    /// could mean not having any connections or needing an array larger than
    /// this one.
    pub fn matmul(
        &mut self,
        mapping: &Mapping,
        weights: &ArrayRef2<T>,
        activations: &ArrayRef2<T>,
    ) -> Array2<T>
    where
        T: num_traits::Zero + Clone + AddAssign + Mul<Output = T>,
    {
        // The activation rows that will be fed into the array in a single pass.
        // Unused rows will be zeroed during each pass.
        // NOTE: We can only execute as many activation rows at a time as the
        // number of rows in the array. All batches (columns) can be sent at
        // once.
        let mut pass_activations = Array2::<T>::zeros((self.nrows(), activations.ncols()));
        let mut pass_weights = Array2::<T>::zeros(self.elements.raw_dim());

        let batch_size = activations.ncols();
        let output_row_count = weights.nrows();
        // NOTE: Keep in mind that these are not the raw outputs from the array
        // but the transformed final result.
        let mut result = Array2::<T>::zeros((output_row_count, batch_size));

        for pass in mapping {
            // PERF: It may be faster to only fill the indices that aren't
            // touched by the following assignment.
            pass_weights.fill(T::zero());
            pass_weights
                .slice_mut(s![pass.range_y(), pass.range_x()])
                .assign(
                    &weights
                        .slice(s![pass.output_rows.clone(), pass.activation_rows.clone(),])
                        .t(),
                );
            self.set_weights_raw(&pass_weights);

            // PERF: It may be faster to only fill the indices that aren't
            // touched by the following assignment.
            pass_activations.fill(T::zero());
            pass_activations
                .slice_mut(s![pass.range_y(), ..])
                .assign(&activations.slice(s![pass.activation_rows.clone(), ..]));

            let pass_result = self.run(&pass_activations);

            result
                // we want to add all columns for all rows that are configured
                // in the pass.
                .slice_mut(s![pass.output_rows.clone(), ..])
                // we use range_x here for the y coordinate because each output
                // row maps to a column in the systolic array.
                .add_assign(&pass_result.slice(s![pass.range_x(), ..]));
        }

        result
    }
}

/// Prepare a matrix to be executed by a [`SystolicArray`].
pub fn shift_activations<T>(activations: &ArrayRef2<T>) -> Array2<T>
where
    T: num_traits::Zero + Clone,
{
    let padding = activations.nrows() - 1;

    let mut shifted = Array2::<T>::zeros((activations.nrows(), activations.ncols() + padding));
    for y in 0..activations.nrows() {
        let offset = padding - y;
        shifted
            .slice_mut(s![y, offset..offset + activations.ncols()])
            .assign(&activations.row(y));
    }

    shifted
}

/// Decode the raw [`SystolicArray`] output as the expected output of a matrix
/// multiplication.
///
/// Unshifts the columns and transposes the result.
pub fn unshift_output<T>(raw_output: &ArrayRef2<T>) -> Array2<T>
where
    T: num_traits::Zero + Clone,
{
    let padding = raw_output.ncols() - 1;

    let ncols_out = raw_output.nrows() - padding;
    let mut output = Array2::<T>::zeros((raw_output.ncols(), ncols_out));
    for y_out in 0..raw_output.ncols() {
        let offset = padding - y_out;
        output
            .slice_mut(s![y_out, ..])
            .assign(&raw_output.slice(s![offset..offset + ncols_out, y_out]));
    }

    output
}
