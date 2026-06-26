pub mod fault;

/// A logical signal value for a single bit.
#[cfg_attr(feature = "proptest", derive(proptest_derive::Arbitrary))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum Signal {
    /// A logical `0`.
    Low = 0,
    /// A logical `1`.
    High = 1,
    /// An unknown signal value. Commonly represented by `X`.
    #[default]
    Unknown = 2,
}

impl Signal {
    /// Drive two signals on the same wire. A short-circuit connection (high-low
    /// or low-high) will cause the singal to be in an unknown state.
    pub fn join(&self, other: Self) -> Self {
        match (self, other) {
            (Self::Low, Self::Low) => Self::Low,
            (Self::High, Self::High) => Self::High,
            _ => Self::Unknown,
        }
    }

    /// Logical AND.
    ///
    /// AND has a dominant value in `0`. Even if the other signal is `X`, the
    /// result will be `0`.
    pub fn and(&self, other: Self) -> Self {
        match (self, other) {
            (Self::Low, _) | (_, Self::Low) => Self::Low,
            (Self::High, Self::High) => Self::High,
            _ => Self::Unknown,
        }
    }

    /// Logical OR.
    ///
    /// OR has a dominant value in `1`. Even if the other signal is `X`, the
    /// result will be `1`.
    pub fn or(&self, other: Self) -> Self {
        match (self, other) {
            (Self::High, _) | (_, Self::High) => Self::High,
            (Self::Low, Self::Low) => Self::Low,
            _ => Self::Unknown,
        }
    }

    /// Logical XOR.
    ///
    /// Unlike AND and OR, XOR has no dominant value. An unknown input signal
    /// will always cause an unknown output.
    pub fn xor(&self, other: Self) -> Self {
        match (self, other) {
            (Self::Low, Self::High) | (Self::High, Self::Low) => Self::High,
            (Self::Low, Self::Low) | (Self::High, Self::High) => Self::Low,
            _ => Self::Unknown,
        }
    }

    /// Logical NOT.
    ///
    /// Flips the signal value. `X` remains `X`.
    #[doc(alias = "not")]
    pub fn invert(&self) -> Self {
        match self {
            Self::Low => Self::High,
            Self::High => Self::Low,
            Self::Unknown => Self::Unknown,
        }
    }
}

impl std::fmt::Display for Signal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Low => "0",
                Self::High => "1",
                Self::Unknown => "X",
            }
        )
    }
}
