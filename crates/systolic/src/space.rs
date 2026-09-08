use crate::helper::u64_;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArrayConfig {
    /// The number of rows in the array.
    nrows: u64,
    /// The number of columns in the array.
    ncols: u64,
    /// The number of bits in the data type.
    dtype_bits: u8,
}

impl ArrayConfig {
    /// Initialize an array dimension object.
    ///
    /// # Panics
    ///
    /// - If nrows or ncols don't fit inside u64.
    /// - If nrows, ncols or dtype_bits are zero.
    pub fn new(nrows: usize, ncols: usize, dtype_bits: u8) -> Self {
        assert!(nrows > 0);
        assert!(ncols > 0);
        assert!(dtype_bits > 0);

        Self {
            nrows: u64_(nrows),
            ncols: u64_(ncols),
            dtype_bits,
        }
    }

    pub fn nrows(&self) -> u64 {
        self.nrows
    }

    pub fn ncols(&self) -> u64 {
        self.ncols
    }

    pub fn dtype_bits(&self) -> u8 {
        self.dtype_bits
    }
}

/// A [`Space::Context`] that can produce the [`ArrayConfig`] it is built on.
///
/// Lets [`crate::fault::TargetedFault`] index its [`Index2`](crate::Index2)
/// component regardless of what additional context a wrapped fault type
/// needs.
pub trait AsArrayConfig: Copy {
    fn array_config(&self) -> ArrayConfig;
}

impl AsArrayConfig for ArrayConfig {
    fn array_config(&self) -> ArrayConfig {
        *self
    }
}

/// A finite type that maps bijectively onto a dense prefix of the non-negative
/// integers.
///
/// `Context` holds the parameters of the enumeration itself - e.g. array
/// geometry, or which register subset is eligible - as opposed to a sampled
/// value's own data. This split exists because `count` has no `self`: the
/// size of the space must be knowable before any value has been constructed,
/// which is what building a `Picker` requires. Consequently `Context` must
/// be the same for every value produced by one enumeration; embedding it in
/// the value instead would let two values disagree about which space they
/// belong to, making an index's meaning ambiguous.
pub trait Space: Sized {
    type Context: Copy;

    /// The total number of distinct values of this type under the given context.
    fn count(context: Self::Context) -> u64;

    /// Map this value to its unique index in `0..count(context)`.
    fn to_index(&self, context: Self::Context) -> u64;

    /// Reconstruct a value from its index. Panics if `index >= count(context)`.
    fn from_index(index: u64, context: Self::Context) -> Self;
}
