use crate::{signal::Signal, simulation::WireId};

/// An output connection from a gate to a wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub(crate) struct OutputPort {
    pub value: Signal,
    pub wire_id: Option<WireId>,
}

impl OutputPort {
    /// Creates a new output port with the given wire ID.
    pub fn new(wire_id: WireId) -> Self {
        Self {
            value: Signal::default(),
            wire_id: Some(wire_id),
        }
    }

    /// Writes a value to the port, triggering a wire update if the value has
    /// changed.
    pub fn write<F>(&mut self, value: Signal, trigger_wire_update: F)
    where
        F: FnOnce(WireId),
    {
        if self.value == value {
            return;
        }

        self.value = value;
        if let Some(wire_id) = self.wire_id {
            trigger_wire_update(wire_id);
        }
    }
}

impl From<WireId> for OutputPort {
    fn from(wire_id: WireId) -> Self {
        Self::new(wire_id)
    }
}

/// An input connection from a wire to a gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub(crate) struct InputPort {
    pub wire_id: Option<WireId>,
}

impl InputPort {
    pub fn read<R>(&self, read_wire: R) -> Signal
    where
        R: FnOnce(WireId) -> Signal,
    {
        match self.wire_id {
            Some(wire_id) => read_wire(wire_id),
            None => Signal::Unknown,
        }
    }
}
