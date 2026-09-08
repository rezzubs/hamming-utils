#[derive(Debug, PartialEq, Eq, Clone, Copy, Default)]
pub(crate) struct Register<T> {
    value: T,
}

impl<T> Register<T> {
    pub fn read(&self) -> T
    where
        T: Clone,
    {
        self.value.clone()
    }

    pub fn write(&mut self, value: T) {
        self.value = value;
    }
}
