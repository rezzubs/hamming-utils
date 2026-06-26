use std::{collections::HashMap, ops::Range};

use crate::{
    Signal,
    builder::Port,
    components::GateKind,
    simulation::{
        builder::{ComponentSpec, Connection, GenericIo, Io, NamedIo},
        bus::{Bus, BusCreationError},
    },
};

/// A helper trait for converting values into [`Port`] connections.
pub trait IntoPort {
    /// Converts the value into a [`Port`] connection.
    fn into_port(self) -> Port;
}

impl IntoPort for Connection {
    fn into_port(self) -> Port {
        Some(self)
    }
}

// Port to itself.
impl IntoPort for Option<Connection> {
    fn into_port(self) -> Port {
        self
    }
}

/// A connection to a wire.
impl IntoPort for &str {
    fn into_port(self) -> Port {
        Some(Connection::Wire {
            name: self.to_owned(),
        })
    }
}

/// A connection to a wire.
impl IntoPort for String {
    fn into_port(self) -> Port {
        Some(Connection::Wire { name: self })
    }
}

/// A connection to a bus.
impl IntoPort for (&str, usize) {
    fn into_port(self) -> Port {
        Some(Connection::Bus {
            name: self.0.to_owned(),
            index: self.1,
        })
    }
}

/// A connection to a bus.
impl IntoPort for (String, usize) {
    fn into_port(self) -> Port {
        Some(Connection::Bus {
            name: self.0,
            index: self.1,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum IoBuilderKind {
    Input,
    Output,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum IoBuilderMapping {
    Auto,
    Manual(Connection),
}

/// A helper for building singular [`NamedIo`] elements.
///
/// This creates a named single-bit IO connection which is connected to a wire
/// with the same name. The target can be overriden with [`map`](IoBuilder::map)
///
/// Construct with [`input`] or [`output`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IoBuilder {
    kind: IoBuilderKind,
    name: String,
    mapping: IoBuilderMapping,
}

impl IoBuilder {
    /// Target a wire with different name than `self`.
    pub fn map(self, connection: impl Into<Connection>) -> Self {
        Self {
            mapping: IoBuilderMapping::Manual(connection.into()),
            ..self
        }
    }

    /// Build the [`NamedIo`].
    pub fn build(self) -> NamedIo {
        let connection = match self.mapping {
            IoBuilderMapping::Auto => Connection::Wire {
                name: self.name.clone(),
            },
            IoBuilderMapping::Manual(connection) => connection,
        };

        let io = Io::Wire(connection);
        let io = match self.kind {
            IoBuilderKind::Input => GenericIo::Input(io),
            IoBuilderKind::Output => GenericIo::Output(io),
        };

        NamedIo {
            name: self.name,
            io,
        }
    }
}

impl From<IoBuilder> for NamedIo {
    fn from(value: IoBuilder) -> Self {
        value.build()
    }
}

/// A helper for building singular [`NamedIo`] elements.
///
/// Construct with [`input_bus`] or [`output_bus`] and select the target with
/// [`map`](IoBusBuilder::map). Can be inserted into a [`SimulationBuilder`]
/// with [`SimulationBuilder::add_io_fallible`](crate::simulation::builder::SimulationBuilder::add_io_fallible).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IoBusBuilder {
    kind: IoBuilderKind,
    name: String,
    range: Range<usize>,
    mapping: HashMap<usize, Connection>,
}

impl IoBusBuilder {
    /// Map a specific bus index.
    pub fn map(mut self, index: usize, connection: impl IntoPort) -> Self {
        if let Some(connection) = connection.into_port() {
            _ = self.mapping.insert(index, connection);
        } else {
            self.mapping.remove(&index);
        }

        self
    }

    /// Build the [`NamedIo`] element from the current state.
    pub fn build(self) -> Result<NamedIo, BusCreationError> {
        let bus: Bus<Connection> = Bus::new(self.range, &self.mapping)?;

        let io = Io::Bus(bus);
        let io = match self.kind {
            IoBuilderKind::Input => GenericIo::Input(io),
            IoBuilderKind::Output => GenericIo::Output(io),
        };

        Ok(NamedIo {
            name: self.name,
            io,
        })
    }
}

impl TryFrom<IoBusBuilder> for NamedIo {
    type Error = BusCreationError;

    fn try_from(value: IoBusBuilder) -> Result<Self, Self::Error> {
        value.build()
    }
}

/// A multi-bit IO builder that automatically maps all indices to a bus with the same name.
///
/// This can be used as an input for
/// [`SimulationBuilder::add_io`](crate::simulation::builder::SimulationBuilder::add_io).
///
/// Construct with [`input_bus`] or [`output_bus`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IoBusBuilderAuto {
    kind: IoBuilderKind,
    name: String,
    range: Range<usize>,
}

impl IoBusBuilderAuto {
    /// Map a connection explicitly. Note that this will remove the current
    /// "automatic" connections - all connections will need to be mapped
    /// explicitly. This will also make `build` return a `Result` so it needs to
    /// be used with
    /// [`SimulationBuilder::add_io_fallible`](crate::simulation::builder::SimulationBuilder::add_io_fallible).
    pub fn map(self, index: usize, connection: impl IntoPort) -> IoBusBuilder {
        let connection = connection.into_port();

        let mapping = match connection {
            Some(connection) => HashMap::from([(index, connection)]),
            None => HashMap::new(),
        };

        IoBusBuilder {
            mapping,
            kind: self.kind,
            name: self.name,
            range: self.range,
        }
    }

    /// Build the bus IO.
    pub fn build(self) -> NamedIo {
        let items = self
            .range
            .clone()
            .map(|index| {
                Some(Connection::Bus {
                    name: self.name.clone(),
                    index,
                })
            })
            .collect();

        let bus = Bus::from_parts(self.range, items)
            .expect("We mapped the full range so it has to be correct");

        let io = Io::Bus(bus);
        let io = match self.kind {
            IoBuilderKind::Input => GenericIo::Input(io),
            IoBuilderKind::Output => GenericIo::Output(io),
        };

        NamedIo {
            name: self.name,
            io,
        }
    }
}

impl From<IoBusBuilderAuto> for NamedIo {
    fn from(value: IoBusBuilderAuto) -> Self {
        value.build()
    }
}

/// Create a single-bit input.
pub fn input(name: impl Into<String>) -> IoBuilder {
    IoBuilder {
        kind: IoBuilderKind::Input,
        name: name.into(),
        mapping: IoBuilderMapping::Auto,
    }
}

/// Create a multi-bit input.
///
/// See [`IoBusBuilderAuto`] for more details.
pub fn input_bus(name: impl Into<String>, range: Range<usize>) -> IoBusBuilderAuto {
    IoBusBuilderAuto {
        kind: IoBuilderKind::Input,
        name: name.into(),
        range,
    }
}

/// Create a single-bit output.
///
/// See [`IoBuilder`] for more details.
pub fn output(name: impl Into<String>) -> IoBuilder {
    IoBuilder {
        kind: IoBuilderKind::Output,
        name: name.into(),
        mapping: IoBuilderMapping::Auto,
    }
}

/// Create a multi-bit output.
///
/// See [`IoBusBuilderAuto`] for more details.
pub fn output_bus(name: impl Into<String>, range: Range<usize>) -> IoBusBuilderAuto {
    IoBusBuilderAuto {
        kind: IoBuilderKind::Output,
        name: name.into(),
        range,
    }
}

macro_rules! gate_helper {
    ($name:ident, $variant:ident, $kind:ident, $length:literal) => {
        /// Create a logic gate.
        pub fn $name<I, O>(inputs: [I; $length], output: O) -> ComponentSpec
        where
            I: IntoPort,
            O: IntoPort,
        {
            ComponentSpec::$variant {
                kind: GateKind::$kind,
                inputs: inputs.map(|i| i.into_port()),
                output: output.into_port(),
            }
        }
    };
}

gate_helper!(and2, Gate2, And, 2);
gate_helper!(and3, Gate3, And, 3);
gate_helper!(and4, Gate4, And, 4);
gate_helper!(nand2, Gate2, Nand, 2);
gate_helper!(nand3, Gate3, Nand, 3);
gate_helper!(nand4, Gate4, Nand, 4);

gate_helper!(or2, Gate2, Or, 2);
gate_helper!(or3, Gate3, Or, 3);
gate_helper!(or4, Gate4, Or, 4);
gate_helper!(nor2, Gate2, Nor, 2);
gate_helper!(nor3, Gate3, Nor, 3);
gate_helper!(nor4, Gate4, Nor, 4);

gate_helper!(xor2, Gate2, Xor, 2);
gate_helper!(xor3, Gate3, Xor, 3);
gate_helper!(xor4, Gate4, Xor, 4);
gate_helper!(xnor2, Gate2, Xnor, 2);
gate_helper!(xnor3, Gate3, Xnor, 3);
gate_helper!(xnor4, Gate4, Xnor, 4);

/// Create a buffer which relays the input to its output.
pub fn buffer(input: impl IntoPort, output: impl IntoPort) -> ComponentSpec {
    ComponentSpec::Buffer {
        input: input.into_port(),
        output: output.into_port(),
    }
}

/// Create an inverting gate.
pub fn inverter(input: impl IntoPort, output: impl IntoPort) -> ComponentSpec {
    ComponentSpec::Inverter {
        input: input.into_port(),
        output: output.into_port(),
    }
}

/// Create a d-type latch.
pub fn dlatch(data: impl IntoPort, enable: impl IntoPort, output: impl IntoPort) -> ComponentSpec {
    ComponentSpec::DLatch {
        data: data.into_port(),
        enable: enable.into_port(),
        output: output.into_port(),
    }
}

/// Create a 2-input multiplexer.
pub fn mux2(
    a: impl IntoPort,
    b: impl IntoPort,
    select: impl IntoPort,
    output: impl IntoPort,
) -> ComponentSpec {
    ComponentSpec::Mux2 {
        a: a.into_port(),
        b: b.into_port(),
        select: select.into_port(),
        output: output.into_port(),
    }
}

macro_rules! compound_helper {
    ($name:ident, $type_name:ident, $($arg:ident),+) => {
        /// Create a compound logic gate.
        pub fn $name(
            $($arg: impl IntoPort),+,
            output: impl IntoPort,
        ) -> ComponentSpec {
            ComponentSpec::$type_name {
                $($arg: $arg.into_port()),+,

                output: output.into_port(),
            }
        }
    };
}

compound_helper!(aoi21, AndOrInvert21, a, b1, b2);
compound_helper!(aoi22, AndOrInvert22, a1, a2, b1, b2);
compound_helper!(aoi211, AndOrInvert211, a, b, c1, c2);
compound_helper!(aoi221, AndOrInvert221, a, b1, b2, c1, c2);
compound_helper!(aoi222, AndOrInvert222, a1, a2, b1, b2, c1, c2);

compound_helper!(oai21, OrAndInvert21, a, b1, b2);
compound_helper!(oai22, OrAndInvert22, a1, a2, b1, b2);
compound_helper!(oai211, OrAndInvert211, a, b, c1, c2);
compound_helper!(oai221, OrAndInvert221, a, b1, b2, c1, c2);
compound_helper!(oai222, OrAndInvert222, a1, a2, b1, b2, c1, c2);
compound_helper!(oai33, OrAndInvert33, a1, a2, a3, b1, b2, b3);

/// Create a half adder
pub fn half_adder(
    a: impl IntoPort,
    b: impl IntoPort,
    sum: impl IntoPort,
    carry: impl IntoPort,
) -> ComponentSpec {
    ComponentSpec::HalfAdder {
        a: a.into_port(),
        b: b.into_port(),
        sum: sum.into_port(),
        carry: carry.into_port(),
    }
}

/// Create a full adder.
pub fn full_adder(
    a: impl IntoPort,
    b: impl IntoPort,
    carry_in: impl IntoPort,
    output: impl IntoPort,
    carry_out: impl IntoPort,
) -> ComponentSpec {
    ComponentSpec::FullAdder {
        a: a.into_port(),
        b: b.into_port(),
        carry_in: carry_in.into_port(),
        output: output.into_port(),
        carry_out: carry_out.into_port(),
    }
}

/// Helper for creating a `0` signal.
pub fn low() -> Signal {
    Signal::Low
}

/// Helper for creating a `1` signal.
pub fn high() -> Signal {
    Signal::High
}

/// Helper for creating an `X` signal.
pub fn unknown() -> Signal {
    Signal::Unknown
}
