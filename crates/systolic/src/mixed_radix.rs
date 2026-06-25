//! Mixed-radix encoding and decoding.
//!
//! A mixed-radix number is a sequence of digits where each digit position has
//! its own radix (base). This is a generalization of positional numeral systems
//! like binary or decimal.
//!
//! The encoding is little-endian: `elements[0]` is the least significant digit.
//! For example, encoding a 2D index `(y, x)` with radixes `(nrows, ncols)` gives
//! `y + nrows * x`.

/// Errors returned by [`encode`] and [`decode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// An element value equals or exceeds its radix.
    #[error("element {element} at position {index} is out of range for radix {radix}")]
    ElementOutOfRange {
        index: usize,
        element: u64,
        radix: u64,
    },
    /// The encoded value overflowed `u64`.
    #[error("arithmetic overflow")]
    Overflow,
    /// The value to decode is larger than what the radixes can represent.
    #[error("encoded value is too large for the given radixes")]
    ValueTooLarge,
}

/// Encode a sequence of elements into a single `u64` using mixed-radix encoding.
///
/// `elements[i]` must be less than `radixes[i]` for each position. The result is:
///
/// ```text
/// elements[0] + radixes[0]*elements[1] + radixes[0]*radixes[1]*elements[2] + ...
/// ```
///
/// See [module docs](crate::mixed_radix) for details on encoding.
///
/// # Panics
///
/// Panics if any radix is 0.
pub(crate) fn encode<const N: usize>(elements: [u64; N], radixes: [u64; N]) -> Result<u64, Error> {
    let mut acc: u64 = 0;
    let mut weight: u64 = 1;

    for i in 0..N {
        let radix = radixes[i];
        let element = elements[i];

        assert!(radix != 0, "radix at position {i} is zero");
        if element >= radix {
            return Err(Error::ElementOutOfRange {
                index: i,
                element,
                radix,
            });
        }

        acc = weight
            .checked_mul(element)
            .and_then(|v| acc.checked_add(v))
            .ok_or(Error::Overflow)?;

        // Only update weight if there is a next position that will use it.
        if i + 1 < N {
            weight = weight.checked_mul(radix).ok_or(Error::Overflow)?;
        }
    }

    Ok(acc)
}

/// Decode a `u64` back into a sequence of elements using mixed-radix decoding.
///
/// This is the inverse of [`encode`]: given the same radixes, `decode(encode(e,
/// r), r) == Ok(e)` for any valid input.
///
/// See [module docs](crate::mixed_radix) for details on encoding.
///
/// # Panics
///
/// Panics if any radix is 0.
pub(crate) fn decode<const N: usize>(
    mut encoded: u64,
    radixes: [u64; N],
) -> Result<[u64; N], Error> {
    let mut elements = [0u64; N];

    for i in 0..N {
        let radix = radixes[i];

        assert!(radix != 0, "radix at position {i} is zero");

        elements[i] = encoded % radix;
        encoded /= radix;
    }

    if encoded != 0 {
        return Err(Error::ValueTooLarge);
    }

    Ok(elements)
}

#[cfg(test)]
mod tests {
    use super::{Error, decode, encode};
    use proptest::prelude::*;

    #[test]
    fn encode_2d_matches_index2_convention() {
        let nrows = 5u64;
        let ncols = 7u64;
        let y = 3u64;
        let x = 4u64;
        assert_eq!(encode([y, x], [nrows, ncols]), Ok(y + nrows * x));
    }

    #[test]
    fn encode_single_element() {
        assert_eq!(encode([3], [10]), Ok(3));
        assert_eq!(encode([0], [1]), Ok(0));
    }

    #[test]
    #[should_panic]
    fn encode_zero_radix_panics() {
        let _ = encode([0], [0]);
    }

    #[test]
    fn encode_element_out_of_range() {
        assert_eq!(
            encode([5], [5]),
            Err(Error::ElementOutOfRange {
                index: 0,
                element: 5,
                radix: 5
            })
        );
        assert_eq!(
            encode([0, 3], [4, 3]),
            Err(Error::ElementOutOfRange {
                index: 1,
                element: 3,
                radix: 3
            })
        );
    }

    #[test]
    fn decode_reverses_encode() {
        let elements = [3u64, 4u64, 2u64];
        let radixes = [5u64, 7u64, 6u64];
        let encoded = encode(elements, radixes).unwrap();
        assert_eq!(decode(encoded, radixes), Ok(elements));
    }

    #[test]
    #[should_panic]
    fn decode_zero_radix_panics() {
        let _ = decode(0, [0u64]);
    }

    #[test]
    fn decode_value_too_large() {
        // Maximum representable value for radixes [3, 4] is 2 + 3*3 = 11.
        assert_eq!(decode(12, [3u64, 4u64]), Err(Error::ValueTooLarge));
    }

    #[test]
    fn empty_encodes_to_zero() {
        assert_eq!(encode([], []), Ok(0));
        assert_eq!(decode(0, []), Ok([]));
        assert_eq!(decode(1, []), Err(Error::ValueTooLarge));
    }

    proptest! {
        #[test]
        fn round_trip_2(
            r0 in 1u64..=16,
            r1 in 1u64..=16,
            e0 in 0u64..16,
            e1 in 0u64..16,
        ) {
            let radixes = [r0, r1];
            let elements = [e0 % r0, e1 % r1];
            let encoded = encode(elements, radixes).unwrap();
            prop_assert_eq!(decode(encoded, radixes), Ok(elements));
        }

        #[test]
        fn round_trip_4(
            r0 in 1u64..=8,
            r1 in 1u64..=8,
            r2 in 1u64..=8,
            r3 in 1u64..=8,
            e0 in 0u64..8,
            e1 in 0u64..8,
            e2 in 0u64..8,
            e3 in 0u64..8,
        ) {
            let radixes = [r0, r1, r2, r3];
            let elements = [e0 % r0, e1 % r1, e2 % r2, e3 % r3];
            let encoded = encode(elements, radixes).unwrap();
            prop_assert_eq!(decode(encoded, radixes), Ok(elements));
        }
    }
}
