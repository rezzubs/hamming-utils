use super::{ComponentId, WireId, bus::Bus};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum Io<Id> {
    Wire(Id),
    Bus(Bus<Id>),
}

pub(crate) type Input = Io<ComponentId>;
pub(crate) type Output = Io<WireId>;
