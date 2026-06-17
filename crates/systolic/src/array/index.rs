use crate::helper::{u64, u64_};
use ndarray::{NdIndex, prelude::*};

/// The type for [`SystolicArray`] dimensions.
pub type Index = u16;

/// An index into a [`SystolicArray`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Index2 {
    pub x: Index,
    pub y: Index,
}

impl Index2 {
    /// Returns an ID for this index, given the array size.
    ///
    /// Returns None on overflow
    ///
    /// # Panics
    ///
    /// - If index is out of bounds for the given dimensions.
    /// - If either dimension is zero.
    pub fn id(&self, array_nrows: usize, array_ncols: usize) -> Option<u64> {
        assert!(array_nrows > 0);
        assert!(array_ncols > 0);

        let array_nrows = u64_(array_nrows);
        let array_ncols = u64_(array_ncols);

        let x = u64_(self.x);
        let y = u64_(self.y);

        if x >= array_ncols {
            panic!("column index out of bounds");
        }
        if y >= array_nrows {
            panic!("row index out of bounds");
        }

        y.checked_add(array_nrows.checked_mul(x)?)
    }

    /// Returns the maximum ID value for a given array size.
    ///
    /// Return None on overflow
    ///
    /// # Panics
    ///
    /// - If either dimension is 0.
    pub fn id_radix(array_nrows: usize, array_ncols: usize) -> Option<u64> {
        assert!(array_nrows > 0);
        assert!(array_ncols > 0);

        u64_(array_nrows).checked_mul(u64_(array_ncols))
    }

    /// Converts an ID back into an , if possible.
    ///
    /// # Panics
    ///
    /// - If either dimension is 0.
    /// - If the ID is out of bounds for the given array size.
    pub fn from_id(id: u64, array_nrows: usize, array_ncols: usize) -> Option<Self> {
        assert!(array_nrows > 0);
        assert!(array_ncols > 0);

        let array_nrows = u64_(array_nrows);
        let array_ncols = u64_(array_ncols);

        // Already checked for zero
        let x_id = id / array_nrows;
        let y_id = id % array_nrows;

        let x = u16::try_from(x_id).ok()?;
        let y = u16::try_from(y_id).ok()?;

        if u64(y) >= array_nrows {
            return None;
        }
        if u64(x) >= array_ncols {
            return None;
        }

        Some(Self { x, y })
    }
}

// SAFETY: Delegates to the safe implementation of (usize, usize).
unsafe impl NdIndex<Ix2> for Index2 {
    fn index_checked(&self, dim: &Ix2, strides: &Ix2) -> Option<isize> {
        (self.y as usize, self.x as usize).index_checked(dim, strides)
    }

    fn index_unchecked(&self, strides: &Ix2) -> isize {
        (self.y as usize, self.x as usize).index_unchecked(strides)
    }
}
