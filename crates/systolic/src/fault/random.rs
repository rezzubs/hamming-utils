use crate::Index2;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use std::ops::{Add, BitXor, Mul};

use crate::array::PeHook;

/// A stochastic XOR-mask fault. Not enumerable; reproduced by re-seeding the
/// same RNG before constructing [`XorMaskHook`].
pub struct RandomFault<T> {
    pub target: Index2,
    /// Candidate (mask, weight) pairs. The mask is XOR-ed into the result; the
    /// weight controls how often each mask is chosen.
    pub entries: Box<[(T, f64)]>,
}

/// A logic fault hook for one PE: every time that PE computes a
/// multiply-add, this randomly draws one of `masks` (weighted by
/// `distribution`) and XORs it into the result, modeling a random bit-level
/// corruption rather than a fixed stuck-at bit.
///
/// Batching multiple examples into one array run, instead of running them
/// one at a time, doesn't bias which examples get corrupted or how. Every
/// value passes through a given PE exactly once as it flows through the
/// array, and the padding cycles added between batched examples produce
/// results that get discarded before they can reach a real output - they
/// only spend RNG draws, they don't leak into anything real.
///
/// This relies on the hook drawing independently each time, with no memory
/// of earlier cycles. A fault model whose corruption chance depends on
/// cycle history wouldn't have this property.
///
/// Note: this is about the *distribution* of outcomes, not exact values -
/// for a fixed seed, batched and unbatched runs consume RNG draws in a
/// different order and will disagree on the precise corrupted output. That
/// disagreement is expected.
pub struct XorMaskHook<T, R> {
    target: Index2,
    masks: Box<[T]>,
    distribution: WeightedIndex<f64>,
    rng: R,
}

impl<T, R: rand::Rng> XorMaskHook<T, R> {
    pub fn from_fault(fault: RandomFault<T>, rng: R) -> Result<Self, rand::distr::weighted::Error> {
        let entries = Vec::from(fault.entries);
        let (masks_vec, weights_vec): (Vec<T>, Vec<f64>) = entries.into_iter().unzip();
        Ok(Self {
            target: fault.target,
            masks: masks_vec.into_boxed_slice(),
            distribution: WeightedIndex::new(weights_vec)?,
            rng,
        })
    }
}

impl<T, R> PeHook<T> for XorMaskHook<T, R>
where
    T: Add<Output = T> + Mul<Output = T> + BitXor<Output = T> + Clone,
    R: rand::Rng,
{
    fn multiply_add(&mut self, index: Index2, activation: T, weight: T, partial_sum: T) -> T
    where
        T: Add<Output = T> + Mul<Output = T>,
    {
        let result = activation * weight + partial_sum;
        if index == self.target {
            let mask_index = self.distribution.sample(&mut self.rng);
            result ^ self.masks[mask_index].clone()
        } else {
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{RandomFault, XorMaskHook};
    use crate::Index2;
    use crate::array::PeHook;
    use rand::SeedableRng;

    fn target() -> Index2 {
        Index2 { x: 0, y: 0 }
    }

    fn other_pe() -> Index2 {
        Index2 { x: 1, y: 0 }
    }

    fn make_hook(mask: u8) -> XorMaskHook<u8, rand::rngs::StdRng> {
        let fault = RandomFault {
            target: target(),
            entries: Box::from([(mask, 1.0_f64)]),
        };
        XorMaskHook::from_fault(fault, rand::rngs::StdRng::seed_from_u64(0))
            .expect("single positive weight must produce a valid distribution")
    }

    #[test]
    fn multiply_add_applies_mask_at_target() {
        let mut hook = make_hook(0b0000_0010);
        // 2 * 3 + 0 = 6 = 0b0000_0110, XOR 0b0000_0010 = 4
        let result: u8 = hook.multiply_add(target(), 2, 3, 0);
        assert_eq!(result, 4);
    }

    #[test]
    fn multiply_add_passes_through_at_non_target() {
        let mut hook = make_hook(0b0000_0010);
        let result: u8 = hook.multiply_add(other_pe(), 2, 3, 0);
        assert_eq!(result, 6);
    }

    #[test]
    fn multiply_add_zero_mask_is_passthrough() {
        let mut hook = make_hook(0);
        let result: u8 = hook.multiply_add(target(), 2, 3, 0);
        assert_eq!(result, 6);
    }
}
