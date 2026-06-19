use crate::Index2;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use std::ops::{Add, BitXor, Mul};

use super::hook::FaultHook;

/// A stochastic XOR-mask fault. Not enumerable; reproduced by re-seeding the
/// same RNG before constructing [`XorMaskHook`].
pub struct RandomFault<T> {
    pub target: Index2,
    /// Candidate (mask, weight) pairs. The mask is XOR-ed into the result; the
    /// weight controls how often each mask is chosen.
    pub entries: Box<[(T, f64)]>,
}

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

impl<T, R> FaultHook<T> for XorMaskHook<T, R>
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
