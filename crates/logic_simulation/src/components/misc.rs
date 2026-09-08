use crate::{
    Signal,
    components::{
        Component,
        port::{InputPort, OutputPort},
    },
    simulation::WireId,
};

/// A D latch that switches its `output` to the `data` value when `enable` is high.
///
/// An [`Signal::Unknown`] value for `enable` will always result in an [`Signal::Unknown`] output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct DLatch {
    pub data: InputPort,
    pub enable: InputPort,
    pub output: OutputPort,
}

impl Component for DLatch {
    fn update<R, T>(&mut self, read_wire: R, trigger_wire_update: T)
    where
        R: Fn(WireId) -> Signal,
        T: FnMut(WireId),
    {
        let output = match self.enable.read(&read_wire) {
            Signal::Unknown => Signal::Unknown,
            Signal::Low => return,
            Signal::High => self.data.read(&read_wire),
        };
        self.output.write(output, trigger_wire_update);
    }

    /// Read the nth output port.
    ///
    /// This component only has a single port, 0 is the only valid value for
    /// `n`.
    fn nth_output(&self, n: u8) -> Option<&OutputPort> {
        match n {
            0 => Some(&self.output),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Mux2 {
    pub a: InputPort,
    pub b: InputPort,
    pub select: InputPort,
    pub output: OutputPort,
}

impl Component for Mux2 {
    fn update<R, T>(&mut self, read_wire: R, trigger_wire_update: T)
    where
        R: Fn(WireId) -> Signal,
        T: FnMut(WireId),
    {
        let output = match self.select.read(&read_wire) {
            Signal::Unknown => Signal::Unknown,
            Signal::Low => self.a.read(read_wire),
            Signal::High => self.b.read(read_wire),
        };

        self.output.write(output, trigger_wire_update);
    }

    fn nth_output(&self, n: u8) -> Option<&OutputPort> {
        match n {
            0 => Some(&self.output),
            _ => None,
        }
    }
}
