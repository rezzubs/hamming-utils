//! Adder components

use crate::{
    Signal,
    components::{
        Component,
        port::{InputPort, OutputPort},
    },
    simulation::WireId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct HalfAdder {
    pub a: InputPort,
    pub b: InputPort,
    pub sum: OutputPort,
    pub carry: OutputPort,
}

impl Component for HalfAdder {
    fn update<R, T>(&mut self, read_wire: R, mut trigger_wire_update: T)
    where
        R: Fn(WireId) -> Signal,
        T: FnMut(WireId),
    {
        let a = self.a.read(&read_wire);
        let b = self.b.read(&read_wire);

        self.sum.write(a.xor(b), &mut trigger_wire_update);
        self.carry.write(a.and(b), &mut trigger_wire_update);
    }

    fn nth_output(&self, n: u8) -> Option<&OutputPort> {
        match n {
            0 => Some(&self.sum),
            1 => Some(&self.carry),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct FullAdder {
    pub a: InputPort,
    pub b: InputPort,
    pub carry_in: InputPort,
    pub sum: OutputPort,
    pub carry_out: OutputPort,
}

impl Component for FullAdder {
    fn update<R, T>(&mut self, read_wire: R, mut trigger_wire_update: T)
    where
        R: Fn(WireId) -> Signal,
        T: FnMut(WireId),
    {
        let a = self.a.read(&read_wire);
        let b = self.b.read(&read_wire);
        let carry_in = self.carry_in.read(&read_wire);

        // Implementation based on FA_X1 in stdcells.v from https://github.com/mflowgen/freepdk-45nm
        let i22 = a.xor(b);
        let sum = carry_in.xor(i22);
        let i16 = a.and(b);
        let i18 = a.or(b);
        let i17 = carry_in.and(i18);
        let carry_out = i16.or(i17);

        self.sum.write(sum, &mut trigger_wire_update);
        self.carry_out.write(carry_out, &mut trigger_wire_update);
    }

    fn nth_output(&self, n: u8) -> Option<&OutputPort> {
        match n {
            0 => Some(&self.sum),
            1 => Some(&self.carry_out),
            _ => None,
        }
    }
}
