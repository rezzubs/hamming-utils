/// A unique identifier for a wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WireId(pub usize);

impl From<usize> for WireId {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

/// A unique identifier for a component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComponentId(pub usize);

impl From<usize> for ComponentId {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

/// A collection of components.
type Components = Vec<Component>;
/// A collection of wires.
type Wires = Vec<Wire>;
