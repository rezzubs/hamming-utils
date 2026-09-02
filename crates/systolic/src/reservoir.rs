use rand::RngExt;

/// A fixed-capacity uniform random sample of a stream, via Algorithm R
/// (reservoir sampling).
///
/// Every item seen so far has equal probability `capacity / seen` of being
/// present in the final sample, without needing to know the stream length in
/// advance.
pub struct Reservoir<T> {
    /// Fixed-length buffer, one slot per capacity unit. Slots past
    /// `seen_count` (while the reservoir isn't full yet) hold `T::default()`
    /// and are never exposed - `samples()` only returns the `..fill()`
    /// prefix, so a placeholder can never be mistaken for a real observation.
    samples: Box<[T]>,
    /// How many items have we *observed* in total. Matches `samples.len()`
    /// only once the reservoir is full.
    observed_count: usize,
}

impl<T: Default> Reservoir<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: std::iter::repeat_with(T::default).take(capacity).collect(),
            observed_count: 0,
        }
    }

    /// Feeds one more item from the stream into the reservoir.
    pub fn observe<R: RngExt>(&mut self, item: T, rng: &mut R) {
        if let Some(slot) = self.samples.get_mut(self.observed_count) {
            *slot = item;
        } else if let Some(slot) = self
            .samples
            .get_mut(rng.random_range(0..=self.observed_count))
        {
            *slot = item;
        }
        self.observed_count += 1;
    }

    /// The currently retained samples.
    pub fn samples(&self) -> &[T] {
        &self.samples[..self.sample_count()]
    }

    pub fn capacity(&self) -> usize {
        self.samples.len()
    }

    /// How many samples are currently retained.
    pub fn sample_count(&self) -> usize {
        self.observed_count.min(self.capacity())
    }

    /// How many items have been observed in total.
    // Not read by chunk 3's `ProfilingArtifact` - it doesn't need a weight,
    // since regime membership is looked up structurally from the mapping -
    // but is kept for spotting a PE with empty support later.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "kept for a future consumer, see comment above")
    )]
    pub fn observed_count(&self) -> usize {
        self.observed_count
    }
}

#[cfg(test)]
mod tests {
    use super::Reservoir;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn fill_from(
        capacity: usize,
        stream: impl IntoIterator<Item = u32>,
        seed: u64,
    ) -> Reservoir<u32> {
        let mut reservoir = Reservoir::new(capacity);
        let mut rng = StdRng::seed_from_u64(seed);
        for item in stream {
            reservoir.observe(item, &mut rng);
        }
        reservoir
    }

    #[test]
    fn fewer_items_than_capacity_keeps_everything() {
        let reservoir = fill_from(5, [10, 20, 30], 0);
        assert_eq!(reservoir.observed_count(), 3);
        assert_eq!(reservoir.sample_count(), 3);
        assert_eq!(reservoir.samples(), [10, 20, 30]);
    }

    #[test]
    fn more_items_than_capacity_caps_fill_at_capacity() {
        let reservoir = fill_from(3, 0..100, 0);
        assert_eq!(reservoir.observed_count(), 100);
        assert_eq!(reservoir.sample_count(), 3);
    }

    #[test]
    fn same_seed_is_deterministic() {
        let a = fill_from(4, 0..50, 42);
        let b = fill_from(4, 0..50, 42);
        assert_eq!(a.samples(), b.samples());
    }

    /// Every item observed should have an equal chance of ending up as the
    /// survivor in a capacity-1 reservoir. Checked by running many independent
    /// trials over the same stream and comparing how often each item actually
    /// wins against how often it should win by chance.
    #[test]
    fn uniform_over_a_known_stream() {
        let item_count = 5u32;
        let trial_count = 20_000u32;
        let mut win_counts = vec![0u32; item_count as usize];
        let mut rng = StdRng::seed_from_u64(0);

        for _ in 0..trial_count {
            let mut reservoir = Reservoir::new(1);
            for item in 0..item_count {
                reservoir.observe(item, &mut rng);
            }
            win_counts[reservoir.samples()[0] as usize] += 1;
        }

        // Algorithm R guarantees every item has probability `1 / item_count` of
        // winning a capacity-1 reservoir. So across `trial_count` independent
        // trials, one item's win count is a sample from a Binomial(trial_count,
        // survival_probability) distribution: it won't land exactly on its
        // expected value, but it won't stray far either. Bounding the deviation
        // by a multiple of that binomial's own standard deviation - rather
        // than a hand-picked percentage - ties the tolerance to how much spread
        // pure chance actually produces, so this catches a real bias (like the
        // classic reservoir-sampling off-by-one that favors some items over
        // others) without being flaky on ordinary noise.
        let survival_probability = 1.0 / f64::from(item_count);
        let expected_win_count = f64::from(trial_count) * survival_probability;
        let standard_deviation =
            (f64::from(trial_count) * survival_probability * (1.0 - survival_probability)).sqrt();
        let allowed_deviation = 5.0 * standard_deviation;

        for win_count in win_counts {
            let deviation = (f64::from(win_count) - expected_win_count).abs();
            assert!(
                deviation <= allowed_deviation,
                "win count {win_count} deviates from expected {expected_win_count} by more than {allowed_deviation}"
            );
        }
    }
}
