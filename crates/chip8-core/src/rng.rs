//! Minimal deterministic PRNG (xorshift64).
//!
//! Not cryptographic. Chosen for reproducibility: the same seed always
//! yields the same sequence, which makes RL episodes replayable.

pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        // Zero is a fixed point of xorshift; remap it to a non-zero constant.
        Self(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed })
    }

    pub(crate) fn next_u8(&mut self) -> u8 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        // High bits have the best statistical quality.
        (x >> 56) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn take(rng: &mut Rng, n: usize) -> Vec<u8> {
        (0..n).map(|_| rng.next_u8()).collect()
    }

    #[test]
    fn same_seed_gives_same_sequence() {
        assert_eq!(take(&mut Rng::new(42), 16), take(&mut Rng::new(42), 16));
    }

    #[test]
    fn different_seeds_give_different_sequences() {
        assert_ne!(take(&mut Rng::new(1), 16), take(&mut Rng::new(2), 16));
    }

    #[test]
    fn zero_seed_does_not_get_stuck() {
        assert!(take(&mut Rng::new(0), 16).iter().any(|&b| b != 0));
    }
}
