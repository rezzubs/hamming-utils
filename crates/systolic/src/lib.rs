//! A [`systolic array`](SystolicArray) simulator with support for custom
//! [`mappings`](Mapping).
//!
//! A [`Mapping`] describes how a matrix multiplication is performed on the
//! array. It can also be used to translate faults into operations on the
//! inputs/outputs; this enables much faster fault simulation than simulating
//! the array.

pub mod array;
mod helper;
#[cfg(test)]
pub(crate) mod test_utilities;

pub use array::{
    Connection, CreationError, Index, Index2, InvalidMappingError, Mapping, Pass, SystolicArray,
    shift_activations, unshift_output,
};
