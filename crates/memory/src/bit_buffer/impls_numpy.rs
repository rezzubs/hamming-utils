use crate::{BitBuffer, SizedBitBuffer};
use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::Bound;

fn bit_index_to_array_index<T>(bit_index: usize, shape: &[usize]) -> (Vec<usize>, usize)
where
    T: SizedBitBuffer,
{
    let item_index = bit_index % T::BITS_COUNT;
    let mut flat = bit_index / T::BITS_COUNT;

    let mut array_index = vec![0usize; shape.len()];
    for dim in 0..shape.len() {
        array_index[dim] = flat % shape[dim];
        flat /= shape[dim];
    }

    (array_index, item_index)
}

impl<'py, T> BitBuffer for Bound<'py, PyArrayDyn<T>>
where
    T: numpy::Element + Copy + SizedBitBuffer,
{
    fn bit_count(&self) -> usize {
        self.len() * T::BITS_COUNT
    }

    fn set_1(&mut self, bit_index: usize) {
        let (array_index, item_index) = bit_index_to_array_index::<T>(bit_index, self.shape());
        self.readwrite()
            .get_mut(array_index)
            .expect("out of bounds")
            .set_1(item_index);
    }

    fn set_0(&mut self, bit_index: usize) {
        let (array_index, item_index) = bit_index_to_array_index::<T>(bit_index, self.shape());
        self.readwrite()
            .get_mut(array_index)
            .expect("out of bounds")
            .set_0(item_index);
    }

    fn is_1(&self, bit_index: usize) -> bool {
        let (array_index, item_index) = bit_index_to_array_index::<T>(bit_index, self.shape());
        self.readonly()
            .get(array_index)
            .expect("out of bounds")
            .is_1(item_index)
    }

    fn flip_bit(&mut self, bit_index: usize) {
        let (array_index, item_index) = bit_index_to_array_index::<T>(bit_index, self.shape());
        self.readwrite()
            .get_mut(array_index)
            .expect("out of bounds")
            .flip_bit(item_index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn total_bits(shape: &[usize], bits_per_element: usize) -> usize {
        shape.iter().product::<usize>() * bits_per_element
    }

    #[test]
    fn single_element_1d() {
        assert_eq!(bit_index_to_array_index::<u8>(0, &[1]), (vec![0], 0));
        assert_eq!(bit_index_to_array_index::<u8>(7, &[1]), (vec![0], 7));
    }

    #[test]
    fn multiple_elements_1d() {
        assert_eq!(bit_index_to_array_index::<u8>(0, &[4]), (vec![0], 0));
        assert_eq!(bit_index_to_array_index::<u8>(8, &[4]), (vec![1], 0));
        assert_eq!(bit_index_to_array_index::<u8>(17, &[4]), (vec![2], 1));
        assert_eq!(bit_index_to_array_index::<u8>(31, &[4]), (vec![3], 7));
    }

    #[test]
    fn known_values_2d() {
        // shape [2, 3], u8: first dim changes fastest
        assert_eq!(bit_index_to_array_index::<u8>(0, &[2, 3]), (vec![0, 0], 0));
        assert_eq!(bit_index_to_array_index::<u8>(8, &[2, 3]), (vec![1, 0], 0));
        assert_eq!(bit_index_to_array_index::<u8>(16, &[2, 3]), (vec![0, 1], 0));
        assert_eq!(bit_index_to_array_index::<u8>(24, &[2, 3]), (vec![1, 1], 0));
    }

    #[test]
    fn bijective_1d() {
        let shape = &[7usize];
        let n = total_bits(shape, u8::BITS_COUNT);
        let outputs: HashSet<_> = (0..n)
            .map(|i| bit_index_to_array_index::<u8>(i, shape))
            .collect();
        assert_eq!(outputs.len(), n);
    }

    #[test]
    fn bijective_2d() {
        let shape = &[3usize, 5];
        let n = total_bits(shape, u16::BITS_COUNT);
        let outputs: HashSet<_> = (0..n)
            .map(|i| bit_index_to_array_index::<u16>(i, shape))
            .collect();
        assert_eq!(outputs.len(), n);
    }

    #[test]
    fn bijective_3d() {
        let shape = &[2usize, 3, 4];
        let n = total_bits(shape, u32::BITS_COUNT);
        let outputs: HashSet<_> = (0..n)
            .map(|i| bit_index_to_array_index::<u32>(i, shape))
            .collect();
        assert_eq!(outputs.len(), n);
    }

    #[test]
    fn indices_in_bounds() {
        let shape = &[3usize, 5, 2];
        let n = total_bits(shape, u16::BITS_COUNT);
        for bit_index in 0..n {
            let (array_index, item_index) = bit_index_to_array_index::<u16>(bit_index, shape);
            assert_eq!(array_index.len(), shape.len());
            for (idx, &dim) in array_index.iter().zip(shape.iter()) {
                assert!(*idx < dim);
            }
            assert!(item_index < u16::BITS_COUNT);
        }
    }
}
