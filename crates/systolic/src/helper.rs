/// A helper function to convert to a u64 or panic on failure.
pub(crate) fn u64_<T>(num: T) -> u64
where
    T: TryInto<u64>,
    <T as std::convert::TryInto<u64>>::Error: std::fmt::Debug,
{
    num.try_into().expect("failed to convert to u64")
}

/// A helper function to convert to a u64.
pub(crate) fn u64<T>(num: T) -> u64
where
    T: Into<u64>,
{
    num.into()
}
