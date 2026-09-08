//! Grouped reduction operations.

use crate::{
    Signal,
    components::{
        Component,
        port::{InputPort, OutputPort},
    },
    simulation::WireId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct AndOrInvert21 {
    pub a: InputPort,
    pub b1: InputPort,
    pub b2: InputPort,
    pub output: OutputPort,
}

impl Component for AndOrInvert21 {
    fn update<R, T>(&mut self, read_wire: R, trigger_wire_update: T)
    where
        R: Fn(WireId) -> Signal,
        T: FnMut(WireId),
    {
        let g1 = self.a.read(&read_wire);
        let g2 = self.b1.read(&read_wire).and(self.b2.read(&read_wire));
        let value = g1.or(g2).invert();
        self.output.write(value, trigger_wire_update);
    }

    fn nth_output(&self, n: u8) -> Option<&OutputPort> {
        match n {
            0 => Some(&self.output),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct AndOrInvert22 {
    pub a1: InputPort,
    pub a2: InputPort,
    pub b1: InputPort,
    pub b2: InputPort,
    pub output: OutputPort,
}

impl Component for AndOrInvert22 {
    fn update<R, T>(&mut self, read_wire: R, trigger_wire_update: T)
    where
        R: Fn(WireId) -> Signal,
        T: FnMut(WireId),
    {
        let g1 = self.a1.read(&read_wire).and(self.a2.read(&read_wire));
        let g2 = self.b1.read(&read_wire).and(self.b2.read(&read_wire));
        let value = g1.or(g2).invert();
        self.output.write(value, trigger_wire_update);
    }

    fn nth_output(&self, n: u8) -> Option<&OutputPort> {
        match n {
            0 => Some(&self.output),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct AndOrInvert211 {
    pub a: InputPort,
    pub b: InputPort,
    pub c1: InputPort,
    pub c2: InputPort,
    pub output: OutputPort,
}

impl Component for AndOrInvert211 {
    fn update<R, T>(&mut self, read_wire: R, trigger_wire_update: T)
    where
        R: Fn(WireId) -> Signal,
        T: FnMut(WireId),
    {
        let g1 = self.a.read(&read_wire);
        let g2 = self.b.read(&read_wire);
        let g3 = self.c1.read(&read_wire).and(self.c2.read(&read_wire));
        let value = g1.or(g2).or(g3).invert();
        self.output.write(value, trigger_wire_update);
    }

    fn nth_output(&self, n: u8) -> Option<&OutputPort> {
        match n {
            0 => Some(&self.output),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct AndOrInvert221 {
    pub a: InputPort,
    pub b1: InputPort,
    pub b2: InputPort,
    pub c1: InputPort,
    pub c2: InputPort,
    pub output: OutputPort,
}

impl Component for AndOrInvert221 {
    fn update<R, T>(&mut self, read_wire: R, trigger_wire_update: T)
    where
        R: Fn(WireId) -> Signal,
        T: FnMut(WireId),
    {
        let g1 = self.a.read(&read_wire);
        let g2 = self.b1.read(&read_wire).and(self.b2.read(&read_wire));
        let g3 = self.c1.read(&read_wire).and(self.c2.read(&read_wire));
        let value = g1.or(g2).or(g3).invert();
        self.output.write(value, trigger_wire_update);
    }

    fn nth_output(&self, n: u8) -> Option<&OutputPort> {
        match n {
            0 => Some(&self.output),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct AndOrInvert222 {
    pub a1: InputPort,
    pub a2: InputPort,
    pub b1: InputPort,
    pub b2: InputPort,
    pub c1: InputPort,
    pub c2: InputPort,
    pub output: OutputPort,
}

impl Component for AndOrInvert222 {
    fn update<R, T>(&mut self, read_wire: R, trigger_wire_update: T)
    where
        R: Fn(WireId) -> Signal,
        T: FnMut(WireId),
    {
        let g1 = self.a1.read(&read_wire).and(self.a2.read(&read_wire));
        let g2 = self.b1.read(&read_wire).and(self.b2.read(&read_wire));
        let g3 = self.c1.read(&read_wire).and(self.c2.read(&read_wire));
        let value = g1.or(g2).or(g3).invert();
        self.output.write(value, trigger_wire_update);
    }

    fn nth_output(&self, n: u8) -> Option<&OutputPort> {
        match n {
            0 => Some(&self.output),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct OrAndInvert21 {
    pub a: InputPort,
    pub b1: InputPort,
    pub b2: InputPort,
    pub output: OutputPort,
}

impl Component for OrAndInvert21 {
    fn update<R, T>(&mut self, read_wire: R, trigger_wire_update: T)
    where
        R: Fn(WireId) -> Signal,
        T: FnMut(WireId),
    {
        let g1 = self.a.read(&read_wire);
        let g2 = self.b1.read(&read_wire).or(self.b2.read(&read_wire));
        let value = g1.and(g2).invert();
        self.output.write(value, trigger_wire_update);
    }

    fn nth_output(&self, n: u8) -> Option<&OutputPort> {
        match n {
            0 => Some(&self.output),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct OrAndInvert22 {
    pub a1: InputPort,
    pub a2: InputPort,
    pub b1: InputPort,
    pub b2: InputPort,
    pub output: OutputPort,
}

impl Component for OrAndInvert22 {
    fn update<R, T>(&mut self, read_wire: R, trigger_wire_update: T)
    where
        R: Fn(WireId) -> Signal,
        T: FnMut(WireId),
    {
        let g1 = self.a1.read(&read_wire).or(self.a2.read(&read_wire));
        let g2 = self.b1.read(&read_wire).or(self.b2.read(&read_wire));
        let value = g1.and(g2).invert();
        self.output.write(value, trigger_wire_update);
    }

    fn nth_output(&self, n: u8) -> Option<&OutputPort> {
        match n {
            0 => Some(&self.output),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct OrAndInvert211 {
    pub a: InputPort,
    pub b: InputPort,
    pub c1: InputPort,
    pub c2: InputPort,
    pub output: OutputPort,
}

impl Component for OrAndInvert211 {
    fn update<R, T>(&mut self, read_wire: R, trigger_wire_update: T)
    where
        R: Fn(WireId) -> Signal,
        T: FnMut(WireId),
    {
        let g1 = self.a.read(&read_wire);
        let g2 = self.b.read(&read_wire);
        let g3 = self.c1.read(&read_wire).or(self.c2.read(&read_wire));
        let value = g1.and(g2).and(g3).invert();
        self.output.write(value, trigger_wire_update);
    }

    fn nth_output(&self, n: u8) -> Option<&OutputPort> {
        match n {
            0 => Some(&self.output),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct OrAndInvert221 {
    pub a: InputPort,
    pub b1: InputPort,
    pub b2: InputPort,
    pub c1: InputPort,
    pub c2: InputPort,
    pub output: OutputPort,
}

impl Component for OrAndInvert221 {
    fn update<R, T>(&mut self, read_wire: R, trigger_wire_update: T)
    where
        R: Fn(WireId) -> Signal,
        T: FnMut(WireId),
    {
        let g1 = self.a.read(&read_wire);
        let g2 = self.b1.read(&read_wire).or(self.b2.read(&read_wire));
        let g3 = self.c1.read(&read_wire).or(self.c2.read(&read_wire));
        let value = g1.and(g2).and(g3).invert();
        self.output.write(value, trigger_wire_update);
    }

    fn nth_output(&self, n: u8) -> Option<&OutputPort> {
        match n {
            0 => Some(&self.output),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct OrAndInvert33 {
    pub a1: InputPort,
    pub a2: InputPort,
    pub a3: InputPort,
    pub b1: InputPort,
    pub b2: InputPort,
    pub b3: InputPort,
    pub output: OutputPort,
}

impl Component for OrAndInvert33 {
    fn update<R, T>(&mut self, read_wire: R, trigger_wire_update: T)
    where
        R: Fn(WireId) -> Signal,
        T: FnMut(WireId),
    {
        let g1 = self
            .a1
            .read(&read_wire)
            .or(self.a2.read(&read_wire))
            .or(self.a3.read(&read_wire));
        let g2 = self
            .b1
            .read(&read_wire)
            .or(self.b2.read(&read_wire))
            .or(self.b3.read(&read_wire));
        let value = g1.and(g2).invert();
        self.output.write(value, trigger_wire_update);
    }

    fn nth_output(&self, n: u8) -> Option<&OutputPort> {
        match n {
            0 => Some(&self.output),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct OrAndInvert222 {
    pub a1: InputPort,
    pub a2: InputPort,
    pub b1: InputPort,
    pub b2: InputPort,
    pub c1: InputPort,
    pub c2: InputPort,
    pub output: OutputPort,
}

impl Component for OrAndInvert222 {
    fn update<R, T>(&mut self, read_wire: R, trigger_wire_update: T)
    where
        R: Fn(WireId) -> Signal,
        T: FnMut(WireId),
    {
        let g1 = self.a1.read(&read_wire).or(self.a2.read(&read_wire));
        let g2 = self.b1.read(&read_wire).or(self.b2.read(&read_wire));
        let g3 = self.c1.read(&read_wire).or(self.c2.read(&read_wire));
        let value = g1.and(g2).and(g3).invert();
        self.output.write(value, trigger_wire_update);
    }

    fn nth_output(&self, n: u8) -> Option<&OutputPort> {
        match n {
            0 => Some(&self.output),
            _ => None,
        }
    }
}
