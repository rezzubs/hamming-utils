use crate::helper::u64;
use crate::id::{ArrayConfig, Space};
use crate::mixed_radix;
use ndarray::{NdIndex, prelude::*};

/// The type for [`crate::SystolicArray`] dimensions.
pub type Index = u16;

/// An index into a [`crate::SystolicArray`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Index2 {
    pub x: Index,
    pub y: Index,
}

impl Space for Index2 {
    type Context = ArrayConfig;

    fn count(config: ArrayConfig) -> u64 {
        config.nrows() * config.ncols()
    }

    fn to_index(&self, config: ArrayConfig) -> u64 {
        let x = u64(self.x);
        let y = u64(self.y);

        mixed_radix::encode([x, y], [config.ncols(), config.nrows()])
            .expect("Index2 coordinates must be within array bounds")
    }

    fn from_index(index: u64, config: ArrayConfig) -> Self {
        let [x, y] = mixed_radix::decode(index, [config.ncols(), config.nrows()])
            .expect("index must be in 0..count");
        let x = u16::try_from(x).expect("x coordinate must fit in u16");
        let y = u16::try_from(y).expect("y coordinate must fit in u16");

        Self { x, y }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::Space;

    fn config(nrows: usize, ncols: usize) -> ArrayConfig {
        ArrayConfig::new(nrows, ncols, 8)
    }

    #[test]
    fn to_index_matches_formula() {
        // index = x + ncols * y
        let cfg = config(5, 7);
        let element_index = Index2 { x: 3, y: 4 };
        assert_eq!(element_index.to_index(cfg), 3 + 7 * 4);
    }

    #[test]
    fn round_trip() {
        let cfg = config(5, 7);
        for y in 0..5u16 {
            for x in 0..7u16 {
                let element_index = Index2 { x, y };
                assert_eq!(Index2::from_index(element_index.to_index(cfg), cfg), element_index);
            }
        }
    }

    #[test]
    fn count_equals_total_elements() {
        let cfg = config(5, 7);
        assert_eq!(Index2::count(cfg), 35);
    }

    #[test]
    #[should_panic]
    fn out_of_bounds_panics() {
        let cfg = config(5, 7);
        let _ = Index2 { x: 7, y: 0 }.to_index(cfg);
    }
}
