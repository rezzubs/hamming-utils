//! Types and functions related to constructing a [`Simulation`]. See
//! [`Simulation::builder`] for an overview.
//!
//! # Helper functions
//!
//! This module defines various helper functions that assist in creating the
//! inputs for [`SimulationBuilder`] methods.
//!
//! ## IO
//!
//! These functions assist in constructing [`NamedIo`] instances for use in.
//!
//! - [`input`], [`input_bus`]
//! - [`output`], [`output_bus`]
//!
//! ## Simple logic gates
//!
//! - [`inverter`], [`buffer`]
//! - [`and2`], [`and3`], [`and4`]
//! - [`or2`], [`or3`], [`or4`]
//! - [`xor2`], [`xor3`], [`xor4`]
//! - [`nand2`], [`nand3`], [`nand4`]
//! - [`nor2`], [`nor3`], [`nor4`]
//! - [`xor2`], [`xor3`], [`xor4`]
//!
//! ## Compound logic gates
//!
//! See [`ComponentSpec#Compound`](ComponentSpec#compound).
//!
//! - [`aoi21`], [`aoi22`], [`aoi211`], [`aoi221`], [`aoi222`]
//! - [`oai21`], [`oai22`], [`oai211`], [`oai221`], [`oai222`], [`oai33`]
//!
//! ## Signals
//!
//! - [`high`]
//! - [`low`]
//! - [`unknown`]

mod helpers;

use std::{
    collections::{HashMap, HashSet, hash_map::Entry as HashMapEntry},
    fmt::Display,
    marker::PhantomData,
    ops::Range,
};

use crate::{
    Signal,
    components::{
        self, AndOrInvert21, AndOrInvert22, AndOrInvert211, AndOrInvert221, AndOrInvert222, Buffer,
        DLatch, FullAdder, Gate, GenericComponent, HalfAdder, Inverter, Mux2, OrAndInvert21,
        OrAndInvert22, OrAndInvert33, OrAndInvert211, OrAndInvert221, OrAndInvert222,
        port::{InputPort, OutputPort},
    },
    simulation::{
        ComponentId, Simulation, WireId,
        bus::Bus,
        io::{Input, Output},
        wire::{OutputLink, Wire},
    },
};

pub use crate::components::GateKind;
pub use helpers::{
    IntoPort, IoBuilder, IoBusBuilder, IoBusBuilderAuto, and2, and3, and4, aoi21, aoi22, aoi211,
    aoi221, aoi222, buffer, dlatch, full_adder, half_adder, high, input, input_bus, inverter, low,
    mux2, nand2, nand3, nand4, nor2, nor3, nor4, oai21, oai22, oai33, oai211, oai221, oai222, or2,
    or3, or4, output, output_bus, unknown, xnor2, xnor3, xnor4, xor2, xor3, xor4,
};

/// Represents an element kind in error reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementKind {
    /// An input element
    Input,
    /// An output element
    Output,
    /// A primitive component
    Component,
}

impl Display for ElementKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ElementKind::Input => write!(f, "input"),
            ElementKind::Output => write!(f, "output"),
            ElementKind::Component => write!(f, "component"),
        }
    }
}

/// An error that was encountered during building.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BuildError {
    /// An element tried to connect to an explicitly defined bus at an index that is out of bounds.
    #[error(
        "{element_kind} {element_name} tried to connect to bus `{bus_name}` at index {index} but the bus has a range of {range:?}"
    )]
    BusIndexOutOfBounds {
        /// The name of the component/input/output that triggered the error.
        element_name: String,
        /// The kind of the element that triggered the error.
        element_kind: ElementKind,
        /// The bus that was attempted to be accessed.
        bus_name: String,
        /// The index that was out of bounds.
        index: usize,
        /// The actual bus range.
        range: Range<usize>,
    },
    /// Tried to assign a value to a bus index that doesn't exist.
    #[error("tried to assign a value to a wire `{name}` that doesn't exist")]
    AssignToUndefinedWire {
        /// The name of the wire that was attempted to be assigned to, but was not found.
        name: String,
    },
    /// Tried to assign a value to a bus that doesn't exist.
    #[error("tried to assign a value to a bus `{name}` that doesn't exist")]
    AssignToUndefinedBus {
        /// The name of the bus that was attempted to be assigned to, but was not found.
        name: String,
    },
    /// Tried to assign a value to a bus but that bit is not connected to anything else.
    #[error(
        "tried to assign a value to a bus {name}[{index}] but that bit is not connected to any components"
    )]
    AssignToUndefinedBusIndex {
        /// The name of the bus that was assigned to.
        name: String,
        /// The bit that was assigned to.
        index: usize,
    },
}

/// A specification for a component.
///
/// # Compound
///
/// Complex gates such as the `AndOrInvert*` and `OrAndInvert*` variants combine
/// two operations. AndOrInvert applies AND to the inner groups such as `a1 and
/// a2` and `b1 and b2`. The results are then ORed and inverted in the end.
/// `OrAndInvert` is analogous.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ComponentSpec {
    /// A gate with two inputs
    Gate2 {
        /// The logic operation.
        kind: GateKind,
        /// The input ports.
        inputs: [Port; 2],
        /// The output port.
        output: Port,
    },
    /// A gate with three inputs
    Gate3 {
        /// The logic operation.
        kind: GateKind,
        /// The input ports.
        inputs: [Port; 3],
        /// The input port.
        output: Port,
    },
    /// A gate with three inputs
    Gate4 {
        /// The logic operation.
        kind: GateKind,
        /// The input ports.
        inputs: [Port; 4],
        /// The input port.
        output: Port,
    },
    /// See [`ComponentSpec#Compound`](ComponentSpec#compound).
    AndOrInvert21 {
        /// Intput port a
        a: Port,
        /// Intput port b1
        b1: Port,
        /// Intput port b2
        b2: Port,
        /// Output port.
        output: Port,
    },
    /// See [`ComponentSpec#Compound`](ComponentSpec#compound).
    AndOrInvert22 {
        /// Input port a1.
        a1: Port,
        /// Input port a2.
        a2: Port,
        /// Input port b1.
        b1: Port,
        /// Input port b2.
        b2: Port,
        /// Output port.
        output: Port,
    },
    /// See [`ComponentSpec#Compound`](ComponentSpec#compound).
    AndOrInvert211 {
        /// Input port a.
        a: Port,
        /// Input port b.
        b: Port,
        /// Input port c1.
        c1: Port,
        /// Input port c2.
        c2: Port,
        /// Output port.
        output: Port,
    },
    /// See [`ComponentSpec#Compound`](ComponentSpec#compound).
    AndOrInvert221 {
        /// Input port a.
        a: Port,
        /// Input port b1.
        b1: Port,
        /// Input port b2.
        b2: Port,
        /// Input port c1.
        c1: Port,
        /// Input port c2.
        c2: Port,
        /// Output port.
        output: Port,
    },
    /// See [`ComponentSpec#Compound`](ComponentSpec#compound).
    AndOrInvert222 {
        /// Input port a1.
        a1: Port,
        /// Input port a2.
        a2: Port,
        /// Input port b1.
        b1: Port,
        /// Input port b2.
        b2: Port,
        /// Input port c1.
        c1: Port,
        /// Input port c2.
        c2: Port,
        /// Output port.
        output: Port,
    },
    /// See [`ComponentSpec#Compound`](ComponentSpec#compound).
    OrAndInvert21 {
        /// Input port a.
        a: Port,
        /// Input port b1.
        b1: Port,
        /// Input port b2.
        b2: Port,
        /// Output port.
        output: Port,
    },
    /// See [`ComponentSpec#Compound`](ComponentSpec#compound).
    OrAndInvert22 {
        /// Input port a1.
        a1: Port,
        /// Input port a2.
        a2: Port,
        /// Input port b1.
        b1: Port,
        /// Input port b2.
        b2: Port,
        /// Output port.
        output: Port,
    },
    /// See [`ComponentSpec#Compound`](ComponentSpec#compound).
    OrAndInvert211 {
        /// Input port a.
        a: Port,
        /// Input port b.
        b: Port,
        /// Input port c1.
        c1: Port,
        /// Input port c2.
        c2: Port,
        /// Output port.
        output: Port,
    },
    /// See [`ComponentSpec#Compound`](ComponentSpec#compound).
    OrAndInvert221 {
        /// Input port a.
        a: Port,
        /// Input port b1.
        b1: Port,
        /// Input port b2.
        b2: Port,
        /// Input port c1.
        c1: Port,
        /// Input port c2.
        c2: Port,
        /// Output port.
        output: Port,
    },
    /// See [`ComponentSpec#Compound`](ComponentSpec#compound).
    OrAndInvert222 {
        /// Input port a1.
        a1: Port,
        /// Input port a2.
        a2: Port,
        /// Input port b1.
        b1: Port,
        /// Input port b2.
        b2: Port,
        /// Input port c1.
        c1: Port,
        /// Input port c2.
        c2: Port,
        /// Output port.
        output: Port,
    },
    /// See [`ComponentSpec#Compound`](ComponentSpec#compound).
    OrAndInvert33 {
        /// Input port a1.
        a1: Port,
        /// Input port a2.
        a2: Port,
        /// Input port a3.
        a3: Port,
        /// Input port b1.
        b1: Port,
        /// Input port b2.
        b2: Port,
        /// Input port b3.
        b3: Port,
        /// Output port.
        output: Port,
    },
    /// Passes the input signal through unchanged.
    Buffer {
        /// Input port.
        input: Port,
        /// Output port.
        output: Port,
    },
    /// Inverts the input signal.
    Inverter {
        /// Input port.
        input: Port,
        /// Output port.
        output: Port,
    },
    /// A transparent D-type latch; captures data while enable is high.
    DLatch {
        /// Data input port.
        data: Port,
        /// Enable input port.
        enable: Port,
        /// Output port.
        output: Port,
    },
    /// A 2-to-1 multiplexer; outputs `a` when select is low, `b` when select is high.
    Mux2 {
        /// Input port a (selected when `select` is low).
        a: Port,
        /// Input port b (selected when `select` is high).
        b: Port,
        /// Select port.
        select: Port,
        /// Output port.
        output: Port,
    },
    /// Adds two single bits, producing a sum and carry-out.
    HalfAdder {
        /// Input port a.
        a: Port,
        /// Input port b.
        b: Port,
        /// Sum output port.
        sum: Port,
        /// Carry output port.
        carry: Port,
    },
    /// Adds two single bits with a carry-in, producing a sum and carry-out.
    FullAdder {
        /// Input port a.
        a: Port,
        /// Input port b.
        b: Port,
        /// Carry input port. Takes the carry from a previous adder.
        carry_in: Port,
        /// Sum output port.
        output: Port,
        /// Carry output port.
        carry_out: Port,
    },
}

/// A connection to a wire or a bit in a bus.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Connection {
    /// A connection to a standalone wire
    Wire {
        /// The name of the wire.
        name: String,
    },
    /// A connection to a single wire in a bus.
    Bus {
        /// The name of the bus.
        name: String,
        /// The index of the wire in the bus.
        index: usize,
    },
}

impl Connection {
    /// Get the name of the wire or bus.
    pub fn name(&self) -> &str {
        match self {
            Connection::Wire { name } => name,
            Connection::Bus { name, .. } => name,
        }
    }
}

impl From<(String, usize)> for Connection {
    fn from((name, index): (String, usize)) -> Self {
        Connection::Bus { name, index }
    }
}

impl From<(&str, usize)> for Connection {
    fn from((name, index): (&str, usize)) -> Self {
        Connection::Bus {
            name: name.to_owned(),
            index,
        }
    }
}

impl From<String> for Connection {
    fn from(value: String) -> Self {
        Connection::Wire { name: value }
    }
}

impl From<&str> for Connection {
    fn from(value: &str) -> Self {
        Connection::Wire {
            name: value.to_owned(),
        }
    }
}

/// An optional connection to a wire or bus.
pub type Port = Option<Connection>;

/// The result of resolving a [`GenericComponent`]'s wire connections to wire IDs.
struct RealizedGeneric<const I: usize, const O: usize> {
    /// The input ports of the component, realized with wire IDs instead of names.
    inputs: [InputPort; I],
    /// The output ports of the component, realized with wire IDs instead of names.
    outputs: [OutputPort; O],
    /// The connected input wires, used to register the component as a wire
    /// consumer. All unique in practice.
    input_connections: Vec<WireId>,
    /// The connected output wires, used to register the component as a wire
    /// producer. All unique in practice.
    output_connections: Vec<WireId>,
}

/// A generic input/output element.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GenericIo {
    /// An input IO connection
    Input(Io),
    /// An output IO connection
    Output(Io),
}

/// A named input or output element.
///
/// The input of [`SimulationBuilder::add_io`] and related functions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NamedIo {
    /// The name of the element
    pub name: String,
    /// The connection.
    pub io: GenericIo,
}

/// The interface of an input/output element. The specific kind is determined by [`GenericIo`]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Io {
    /// A single connection.
    Wire(Connection),
    /// Multiple connections grouped with a [`Bus`] interface.
    Bus(Bus<Connection>),
}

/// A wire or bus. Wires and busses have a shared namespace and are stored using
/// this enum.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Connector {
    Wire,
    Bus {
        /// The range of the bus.
        range: Range<usize>,
    },
}

/// An assignment. Drives the target permanently with a signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Assignment {
    /// Assign to a wire.
    Wire(Signal),
    /// Assign to a bus at an index.
    Bus {
        /// The index in the bus.
        index: usize,
        /// The signal to assign.
        signal: Signal,
    },
}

/// See [`Simulation::builder`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimulationBuilder<F = ()> {
    /// The components that have been configured.
    components: HashMap<String, ComponentSpec>,
    /// The connectors that have been configured.
    connectors: HashMap<String, Connector>,
    /// The IO connections that have been configured.
    ios: HashMap<String, GenericIo>,
    assignments: HashMap<(String, Option<usize>), Signal>,
    fault_kind: PhantomData<F>,
}

impl<F> SimulationBuilder<F> {
    /// Build the simulation.
    pub fn build(self) -> Result<Simulation<F>, BuildError> {
        let mut simulation = RunningBuilder::default().realize(
            self.components,
            self.connectors,
            self.ios,
            self.assignments,
        )?;
        for id in 0..simulation.components.len() {
            simulation.component_updates.insert(ComponentId(id));
        }
        for id in 0..simulation.wires.len() {
            simulation.wire_updates.insert(WireId(id));
        }

        Ok(simulation)
    }

    /// Create a new simulation builder.
    pub fn new() -> Self {
        Self {
            components: HashMap::new(),
            connectors: HashMap::new(),
            ios: HashMap::new(),
            assignments: HashMap::new(),
            fault_kind: PhantomData,
        }
    }

    /// Add a component to the simulation.
    pub fn add_component(
        &mut self,
        name: impl Into<String>,
        component: ComponentSpec,
    ) -> &mut Self {
        self.components.insert(name.into(), component);

        self
    }

    /// Add an IO connection to the simulation.
    ///
    /// See also [`SimulationBuilder::add_io_many`].
    pub fn add_io<Io>(&mut self, io: Io) -> &mut Self
    where
        Io: Into<NamedIo>,
    {
        let io = io.into();

        self.ios.insert(io.name, io.io);

        self
    }

    /// Add an IO connection to the simulation.
    ///
    /// See also [`SimulationBuilder::with_io_many`].
    pub fn with_io<Io>(mut self, io: Io) -> Self
    where
        Io: Into<NamedIo>,
    {
        self.add_io(io);
        self
    }

    /// Add many IO connections.
    pub fn add_io_many<I, Io>(&mut self, ios: I) -> &mut Self
    where
        I: IntoIterator<Item = Io>,
        Io: Into<NamedIo>,
    {
        for io in ios {
            self.add_io(io);
        }

        self
    }

    /// Add many IO connections.
    pub fn with_io_many<I, Io>(mut self, ios: I) -> Self
    where
        I: IntoIterator<Item = Io>,
        Io: Into<NamedIo>,
    {
        self.add_io_many(ios);
        self
    }

    /// Add a component to the simulation.
    pub fn with_component(mut self, name: impl Into<String>, component: ComponentSpec) -> Self {
        self.add_component(name, component);
        self
    }

    /// Add an IO connection to the simulation which may fail such as [`IoBusBuilder`].
    ///
    /// See also [`SimulationBuilder::add_io_many_fallible`].
    pub fn add_io_fallible<Io>(&mut self, io: Io) -> Result<&mut Self, Io::Error>
    where
        Io: TryInto<NamedIo>,
    {
        let io = io.try_into()?;
        self.ios.insert(io.name, io.io);

        Ok(self)
    }

    /// Add an IO connection to the simulation which may fail such as [`IoBusBuilder`].
    ///
    /// See also [`SimulationBuilder::with_io_many_fallible`].
    pub fn with_io_fallible<Io>(mut self, io: Io) -> Result<Self, Io::Error>
    where
        Io: TryInto<NamedIo>,
    {
        self.add_io_fallible(io)?;
        Ok(self)
    }

    /// Add many IO connections to the simulation which may fail such as [`IoBusBuilder`].
    pub fn add_io_many_fallible<I, Io>(&mut self, ios: I) -> Result<&mut Self, Io::Error>
    where
        I: IntoIterator<Item = Io>,
        Io: TryInto<NamedIo>,
    {
        for io in ios {
            self.add_io_fallible(io)?;
        }

        Ok(self)
    }

    /// Add many IO connections to the simulation which may fail such as [`IoBusBuilder`].
    pub fn with_io_many_fallible<I, Io>(mut self, ios: I) -> Result<Self, Io::Error>
    where
        I: IntoIterator<Item = Io>,
        Io: TryInto<NamedIo>,
    {
        self.add_io_many_fallible(ios)?;

        Ok(self)
    }

    /// Add a wire to the simulation.
    ///
    /// Wires are also added implicitly if a component references an undefined
    /// name.
    pub fn add_wire(&mut self, name: impl Into<String>) -> &mut Self {
        self.connectors.insert(name.into(), Connector::Wire);
        self
    }

    /// Add a wire to the simulation.
    ///
    /// Wires are also added implicitly if a component references an undefined
    /// name.
    pub fn with_wire(mut self, name: impl Into<String>) -> Self {
        self.add_wire(name);
        self
    }

    /// Add a bus to the simulation.
    ///
    /// Wires are also added implicitly if a component references an undefined
    /// name. Adding the bus explicitly will put bounds checks on components
    /// which reference that bus. This manifests as
    /// [`BuildError::BusIndexOutOfBounds`].
    pub fn add_bus(&mut self, name: impl Into<String>, range: Range<usize>) -> &mut Self {
        self.connectors
            .insert(name.into(), Connector::Bus { range });
        self
    }

    /// Add a bus to the simulation.
    ///
    /// Wires are also added implicitly if a component references an undefined
    /// name. Adding the bus explicitly will put bounds checks on components
    /// which reference that bus. This manifests as
    /// [`BuildError::BusIndexOutOfBounds`].
    pub fn with_bus(mut self, name: impl Into<String>, range: Range<usize>) -> Self {
        self.add_bus(name, range);
        self
    }

    /// Add a permanent driver to a wire or bus.
    ///
    /// This does not implicitly define a wire. If not other components
    /// reference this connector or if the connector isn't defined explicitly
    /// then the build funciton will error.
    pub fn add_assignment(&mut self, name: impl Into<String>, assignment: Assignment) -> &mut Self {
        let name = name.into();

        // Overrides the previous entry if it exists.
        _ = match assignment {
            Assignment::Wire(signal) => self.assignments.insert((name, None), signal),
            Assignment::Bus { index, signal } => {
                self.assignments.insert((name, Some(index)), signal)
            }
        };

        self
    }

    /// Add a permanent driver to a wire or bus.
    ///
    /// This does not implicitly define a wire. If not other components
    /// reference this connector or if the connector isn't defined explicitly
    /// then the build funciton will error.
    pub fn with_assignment(mut self, name: impl Into<String>, assignment: Assignment) -> Self {
        self.add_assignment(name, assignment);
        self
    }
}

impl Default for SimulationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// How was the bus connector added.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum BuilderBusKind {
    /// The range of this bus is explicitly specified.
    Explicit,
    /// We don't know the actual range of this bus but we have inferred the
    /// minimum length.
    Inferred,
}

/// A representation of a bus connector in the building process.
#[derive(Debug, Clone, PartialEq, Eq)]
struct BuilderBus {
    /// What is the range of this bus. Usage depends on `kind`.
    range: Range<usize>,
    /// Maps bus indices to wire IDs.
    wires: HashMap<usize, WireId>,
    /// How was the bus connector added.
    kind: BuilderBusKind,
}

/// Data that's being built into a [`Simulation`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct RunningBuilder {
    /// The list of wires that will be inserted into the [`Simulation`].
    wires: Vec<Wire>,
    /// The list of components that will be inserted into the [`Simulation`].
    components: Vec<GenericComponent>,
    /// Map of inputs that will be inserted into the [`Simulation`].
    inputs: HashMap<String, Input>,
    /// Map of outputs that will be inserted into the [`Simulation`].
    outputs: HashMap<String, Output>,

    // The following are only used during building.
    /// Maps wire names to IDs.
    wire_map: HashMap<String, WireId>,
    /// Maps bus names to wire ids and metadata.
    bus_map: HashMap<String, BuilderBus>,
}

impl RunningBuilder {
    /// Retrun the wire ID for this connection, creating the wire if it doesn't exist.
    fn realize_connection(
        &mut self,
        element_name: &str,
        element_kind: ElementKind,
        connection: Connection,
    ) -> Result<WireId, BuildError> {
        // NOTE: We know that the same name can only be used for a wire or a
        // bus, not both.
        Ok(match connection {
            Connection::Wire { name } => match self.wire_map.entry(name) {
                HashMapEntry::Occupied(entry) => *entry.get(),
                HashMapEntry::Vacant(entry) => {
                    self.wires.push(Wire::new());
                    let id = WireId(self.wires.len() - 1);
                    entry.insert(id);
                    id
                }
            },
            Connection::Bus { name, index } => {
                let mut entry = self.bus_map.entry(name.clone());
                let bus = match entry {
                    HashMapEntry::Occupied(ref mut entry) => entry.get_mut(),
                    // This is only executed for implicitly added busses as
                    // explicit connectors are added before components are
                    // evaluated.
                    HashMapEntry::Vacant(vacant_entry) => {
                        let bus = BuilderBus {
                            range: index..index + 1,
                            // Leave the map empty for now, the element will be
                            // inserted at the end of the function.
                            wires: HashMap::new(),
                            kind: BuilderBusKind::Inferred {},
                        };
                        vacant_entry.insert(bus)
                    }
                };

                if !bus.range.contains(&index) {
                    // This branch is only relevant for the occupied entry case
                    match bus.kind {
                        BuilderBusKind::Explicit => {
                            return Err(BuildError::BusIndexOutOfBounds {
                                element_name: element_name.to_owned(),
                                element_kind,
                                bus_name: name,
                                index,
                                range: bus.range.clone(),
                            });
                        }
                        BuilderBusKind::Inferred => {
                            bus.range.end = bus.range.end.max(index + 1);
                            bus.range.start = bus.range.start.min(index);
                        }
                    }
                }

                *bus.wires.entry(index).or_insert_with(|| {
                    self.wires.push(Wire::new());
                    WireId(self.wires.len() - 1)
                })
            }
        })
    }

    /// Get wire ids
    fn realize_generic<const I: usize, const O: usize>(
        &mut self,
        component_name: &str,
        inputs: [Port; I],
        outputs: [Port; O],
    ) -> Result<RealizedGeneric<I, O>, BuildError> {
        let mut input_connections = Vec::<WireId>::new();
        let mut realized_inputs: [InputPort; I] = [InputPort::default(); I];
        for (slot, connection) in realized_inputs.iter_mut().zip(inputs) {
            let wire_id = connection
                .map(|connection| {
                    self.realize_connection(component_name, ElementKind::Component, connection)
                })
                .transpose()?;

            if let Some(wire_id) = wire_id {
                input_connections.push(wire_id);
            }

            slot.wire_id = wire_id;
        }

        let mut output_connections = Vec::<WireId>::new();
        let mut realized_outputs: [OutputPort; O] = [OutputPort::default(); O];
        for (slot, connection) in realized_outputs.iter_mut().zip(outputs) {
            let wire_id = connection
                .map(|connection| {
                    self.realize_connection(component_name, ElementKind::Component, connection)
                })
                .transpose()?;

            if let Some(wire_id) = wire_id {
                output_connections.push(wire_id);
            }

            slot.wire_id = wire_id;
        }

        Ok(RealizedGeneric {
            inputs: realized_inputs,
            outputs: realized_outputs,
            input_connections,
            output_connections,
        })
    }

    /// Add a component to the simulation, realizing its connections to actual
    /// IDs. Additionally creates backwards references for the connected wires.
    fn add_generic_component(&mut self, name: &str, spec: ComponentSpec) -> Result<(), BuildError> {
        let (component, input_connections, output_connections) = match spec {
            ComponentSpec::Gate2 {
                kind,
                inputs,
                output,
            } => {
                let RealizedGeneric {
                    inputs,
                    outputs: [output],
                    input_connections,
                    output_connections,
                } = self.realize_generic(name, inputs, [output])?;
                (
                    GenericComponent::Gate2(Gate {
                        kind,
                        inputs,
                        output,
                    }),
                    input_connections,
                    output_connections,
                )
            }
            ComponentSpec::Gate3 {
                kind,
                inputs,
                output,
            } => {
                let RealizedGeneric {
                    inputs,
                    outputs: [output],
                    input_connections,
                    output_connections,
                } = self.realize_generic(name, inputs, [output])?;
                (
                    GenericComponent::Gate3(Gate {
                        kind,
                        inputs,
                        output,
                    }),
                    input_connections,
                    output_connections,
                )
            }
            ComponentSpec::Gate4 {
                kind,
                inputs,
                output,
            } => {
                let RealizedGeneric {
                    inputs,
                    outputs: [output],
                    input_connections,
                    output_connections,
                } = self.realize_generic(name, inputs, [output])?;
                (
                    GenericComponent::Gate4(Gate {
                        kind,
                        inputs,
                        output,
                    }),
                    input_connections,
                    output_connections,
                )
            }
            ComponentSpec::Inverter { input, output } => {
                let RealizedGeneric {
                    inputs: [input],
                    outputs: [output],
                    input_connections,
                    output_connections,
                } = self.realize_generic(name, [input], [output])?;

                (
                    GenericComponent::Inverter(Inverter::new(input, output)),
                    input_connections,
                    output_connections,
                )
            }
            ComponentSpec::Buffer { input, output } => {
                let RealizedGeneric {
                    inputs: [input],
                    outputs: [output],
                    input_connections,
                    output_connections,
                } = self.realize_generic(name, [input], [output])?;

                (
                    GenericComponent::Buffer(Buffer::new(input, output)),
                    input_connections,
                    output_connections,
                )
            }
            ComponentSpec::DLatch {
                data,
                enable,
                output,
            } => {
                let RealizedGeneric {
                    inputs: [data, enable],
                    outputs: [output],
                    input_connections,
                    output_connections,
                } = self.realize_generic(name, [data, enable], [output])?;

                (
                    GenericComponent::DLatch(DLatch {
                        data,
                        enable,
                        output,
                    }),
                    input_connections,
                    output_connections,
                )
            }
            ComponentSpec::AndOrInvert21 { a, b1, b2, output } => {
                let RealizedGeneric {
                    inputs: [a, b1, b2],
                    outputs: [output],
                    input_connections,
                    output_connections,
                } = self.realize_generic(name, [a, b1, b2], [output])?;

                (
                    GenericComponent::AndOrInvert21(AndOrInvert21 { a, b1, b2, output }),
                    input_connections,
                    output_connections,
                )
            }
            ComponentSpec::AndOrInvert22 {
                a1,
                a2,
                b1,
                b2,
                output,
            } => {
                let RealizedGeneric {
                    inputs: [a1, a2, b1, b2],
                    outputs: [output],
                    input_connections,
                    output_connections,
                } = self.realize_generic(name, [a1, a2, b1, b2], [output])?;

                (
                    GenericComponent::AndOrInvert22(AndOrInvert22 {
                        a1,
                        a2,
                        b1,
                        b2,
                        output,
                    }),
                    input_connections,
                    output_connections,
                )
            }
            ComponentSpec::AndOrInvert211 {
                a,
                b,
                c1,
                c2,
                output,
            } => {
                let RealizedGeneric {
                    inputs: [a, b, c1, c2],
                    outputs: [output],
                    input_connections,
                    output_connections,
                } = self.realize_generic(name, [a, b, c1, c2], [output])?;

                (
                    GenericComponent::AndOrInvert211(AndOrInvert211 {
                        a,
                        b,
                        c1,
                        c2,
                        output,
                    }),
                    input_connections,
                    output_connections,
                )
            }
            ComponentSpec::AndOrInvert221 {
                a,
                b1,
                b2,
                c1,
                c2,
                output,
            } => {
                let RealizedGeneric {
                    inputs: [a, b1, b2, c1, c2],
                    outputs: [output],
                    input_connections,
                    output_connections,
                } = self.realize_generic(name, [a, b1, b2, c1, c2], [output])?;

                (
                    GenericComponent::AndOrInvert221(AndOrInvert221 {
                        a,
                        b1,
                        b2,
                        c1,
                        c2,
                        output,
                    }),
                    input_connections,
                    output_connections,
                )
            }
            ComponentSpec::AndOrInvert222 {
                a1,
                a2,
                b1,
                b2,
                c1,
                c2,
                output,
            } => {
                let RealizedGeneric {
                    inputs: [a1, a2, b1, b2, c1, c2],
                    outputs: [output],
                    input_connections,
                    output_connections,
                } = self.realize_generic(name, [a1, a2, b1, b2, c1, c2], [output])?;

                (
                    GenericComponent::AndOrInvert222(AndOrInvert222 {
                        a1,
                        a2,
                        b1,
                        b2,
                        c1,
                        c2,
                        output,
                    }),
                    input_connections,
                    output_connections,
                )
            }
            ComponentSpec::OrAndInvert21 { a, b1, b2, output } => {
                let RealizedGeneric {
                    inputs: [a, b1, b2],
                    outputs: [output],
                    input_connections,
                    output_connections,
                } = self.realize_generic(name, [a, b1, b2], [output])?;

                (
                    GenericComponent::OrAndInvert21(OrAndInvert21 { a, b1, b2, output }),
                    input_connections,
                    output_connections,
                )
            }
            ComponentSpec::OrAndInvert22 {
                a1,
                a2,
                b1,
                b2,
                output,
            } => {
                let RealizedGeneric {
                    inputs: [a1, a2, b1, b2],
                    outputs: [output],
                    input_connections,
                    output_connections,
                } = self.realize_generic(name, [a1, a2, b1, b2], [output])?;

                (
                    GenericComponent::OrAndInvert22(OrAndInvert22 {
                        a1,
                        a2,
                        b1,
                        b2,
                        output,
                    }),
                    input_connections,
                    output_connections,
                )
            }
            ComponentSpec::OrAndInvert211 {
                a,
                b,
                c1,
                c2,
                output,
            } => {
                let RealizedGeneric {
                    inputs: [a, b, c1, c2],
                    outputs: [output],
                    input_connections,
                    output_connections,
                } = self.realize_generic(name, [a, b, c1, c2], [output])?;

                (
                    GenericComponent::OrAndInvert211(OrAndInvert211 {
                        a,
                        b,
                        c1,
                        c2,
                        output,
                    }),
                    input_connections,
                    output_connections,
                )
            }
            ComponentSpec::OrAndInvert221 {
                a,
                b1,
                b2,
                c1,
                c2,
                output,
            } => {
                let RealizedGeneric {
                    inputs: [a, b1, b2, c1, c2],
                    outputs: [output],
                    input_connections,
                    output_connections,
                } = self.realize_generic(name, [a, b1, b2, c1, c2], [output])?;

                (
                    GenericComponent::OrAndInvert221(OrAndInvert221 {
                        a,
                        b1,
                        b2,
                        c1,
                        c2,
                        output,
                    }),
                    input_connections,
                    output_connections,
                )
            }
            ComponentSpec::OrAndInvert222 {
                a1,
                a2,
                b1,
                b2,
                c1,
                c2,
                output,
            } => {
                let RealizedGeneric {
                    inputs: [a1, a2, b1, b2, c1, c2],
                    outputs: [output],
                    input_connections,
                    output_connections,
                } = self.realize_generic(name, [a1, a2, b1, b2, c1, c2], [output])?;

                (
                    GenericComponent::OrAndInvert222(OrAndInvert222 {
                        a1,
                        a2,
                        b1,
                        b2,
                        c1,
                        c2,
                        output,
                    }),
                    input_connections,
                    output_connections,
                )
            }
            ComponentSpec::OrAndInvert33 {
                a1,
                a2,
                a3,
                b1,
                b2,
                b3,
                output,
            } => {
                let RealizedGeneric {
                    inputs: [a1, a2, a3, b1, b2, b3],
                    outputs: [output],
                    input_connections,
                    output_connections,
                } = self.realize_generic(name, [a1, a2, a3, b1, b2, b3], [output])?;

                (
                    GenericComponent::OrAndInvert33(OrAndInvert33 {
                        a1,
                        a2,
                        a3,
                        b1,
                        b2,
                        b3,
                        output,
                    }),
                    input_connections,
                    output_connections,
                )
            }
            ComponentSpec::Mux2 {
                a,
                b,
                select,
                output,
            } => {
                let RealizedGeneric {
                    inputs: [a, b, select],
                    outputs: [output],
                    input_connections,
                    output_connections,
                } = self.realize_generic(name, [a, b, select], [output])?;

                (
                    GenericComponent::Mux2(Mux2 {
                        a,
                        b,
                        select,
                        output,
                    }),
                    input_connections,
                    output_connections,
                )
            }
            ComponentSpec::HalfAdder {
                a,
                b,
                sum: output,
                carry,
            } => {
                let RealizedGeneric {
                    inputs: [a, b],
                    outputs: [output, carry],
                    input_connections,
                    output_connections,
                } = self.realize_generic(name, [a, b], [output, carry])?;

                (
                    GenericComponent::HalfAdder(HalfAdder {
                        a,
                        b,
                        sum: output,
                        carry,
                    }),
                    input_connections,
                    output_connections,
                )
            }
            ComponentSpec::FullAdder {
                a,
                b,
                carry_in,
                output,
                carry_out,
            } => {
                let RealizedGeneric {
                    inputs: [a, b, carry_in],
                    outputs: [output, carry_out],
                    input_connections,
                    output_connections,
                } = self.realize_generic(name, [a, b, carry_in], [output, carry_out])?;

                (
                    GenericComponent::FullAdder(FullAdder {
                        a,
                        b,
                        carry_in,
                        sum: output,
                        carry_out,
                    }),
                    input_connections,
                    output_connections,
                )
            }
        };

        self.add_component(component, &input_connections, &output_connections);

        Ok(())
    }

    /// Adds an input to the simulation.
    fn add_input(&mut self, name: String, input: Io) -> Result<(), BuildError> {
        match input {
            Io::Wire(connection) => {
                let wire_id = self.realize_connection(&name, ElementKind::Input, connection)?;
                let input_id = self.add_component(
                    GenericComponent::Input(components::OutputPort::new(wire_id)),
                    &[],
                    &[wire_id],
                );

                let previous = self.inputs.insert(name, Input::Wire(input_id));
                debug_assert_eq!(
                    previous, None,
                    "The element list is guaranteed to have unique names."
                )
            }
            Io::Bus(bus) => {
                let (range, connections) = bus.into_inner();

                let wire_ids = connections
                    .into_iter()
                    .map(|maybe_connection| {
                        maybe_connection
                            .map(|connection| {
                                self.realize_connection(&name, ElementKind::Input, connection)
                            })
                            .transpose()
                    })
                    .collect::<Result<Vec<Option<WireId>>, BuildError>>()?;

                let input_ids = wire_ids
                    .into_iter()
                    .map(|maybe_wire_id| {
                        maybe_wire_id.map(|wire_id| {
                            self.add_component(
                                GenericComponent::Input(components::OutputPort::new(wire_id)),
                                &[],
                                &[wire_id],
                            )
                        })
                    })
                    .collect::<Vec<Option<ComponentId>>>();

                let bus = Bus::from_parts(range, input_ids)
                    .expect("The range has already been validated by realize_connection");
                let previous = self.inputs.insert(name, Input::Bus(bus));

                debug_assert_eq!(
                    previous, None,
                    "The element list is guaranteed to have unique names."
                );
            }
        }

        Ok(())
    }

    /// Adds an output to the simulation.
    fn add_output(&mut self, name: String, output: Io) -> Result<(), BuildError> {
        match output {
            Io::Wire(connection) => {
                let wire_id = self.realize_connection(&name, ElementKind::Output, connection)?;
                self.outputs.insert(name, Output::Wire(wire_id));
            }
            Io::Bus(bus) => {
                let (range, items) = bus.into_inner();

                let wire_ids = items
                    .into_iter()
                    .map(|maybe_connection| {
                        maybe_connection
                            .map(|connection| {
                                self.realize_connection(&name, ElementKind::Output, connection)
                            })
                            .transpose()
                    })
                    .collect::<Result<Vec<Option<WireId>>, BuildError>>()?;

                let bus =
                    Bus::from_parts(range, wire_ids).expect("The range has already been validated");
                self.outputs.insert(name, Output::Bus(bus));
            }
        }

        Ok(())
    }

    /// Add a component to the simulation and set the connections on the relevant wires.
    fn add_component(
        &mut self,
        component: GenericComponent,
        input_connections: &[WireId],
        output_connections: &[WireId],
    ) -> ComponentId {
        self.components.push(component);
        let id = ComponentId(self.components.len() - 1);

        for (output_index, wire_id) in output_connections.iter().enumerate() {
            self.wires[wire_id.0].link_output(OutputLink {
                component_id: id,
                output_index: u8::try_from(output_index)
                    .expect("we don't have any components with more than u8::MAX outputs"),
            });
        }
        for wire_id in input_connections {
            self.wires[wire_id.0].link_input(id);
        }

        id
    }

    fn add_assignment(&mut self, name: &str, assignment: Assignment) -> Result<(), BuildError> {
        match assignment {
            Assignment::Wire(signal) => {
                let Some(&wire_id) = self.wire_map.get(name) else {
                    return Err(BuildError::AssignToUndefinedWire {
                        name: name.to_owned(),
                    });
                };

                self.add_component(
                    GenericComponent::Input(components::OutputPort {
                        wire_id: Some(wire_id),
                        value: signal,
                    }),
                    &[],
                    &[wire_id],
                );
            }
            Assignment::Bus { index, signal } => {
                let Some(bus) = self.bus_map.get(name) else {
                    return Err(BuildError::AssignToUndefinedBus {
                        name: name.to_owned(),
                    });
                };

                let Some(&wire_id) = bus.wires.get(&index) else {
                    return Err(BuildError::AssignToUndefinedBusIndex {
                        name: name.to_owned(),
                        index,
                    });
                };

                self.add_component(
                    GenericComponent::Input(components::OutputPort {
                        wire_id: Some(wire_id),
                        value: signal,
                    }),
                    &[],
                    &[wire_id],
                );
            }
        }

        Ok(())
    }

    /// Maps all names to concrete IDs and builds the simulation.
    fn realize<F>(
        mut self,
        components: HashMap<String, ComponentSpec>,
        connectors: HashMap<String, Connector>,
        ios: HashMap<String, GenericIo>,
        assignments: HashMap<(String, Option<usize>), Signal>,
    ) -> Result<Simulation<F>, BuildError> {
        // `realize_connection` expects connectors to be added before components and IOs.
        for (name, connector) in connectors {
            match connector {
                Connector::Wire => {
                    self.wires.push(Wire::new());
                    let id = WireId(self.wires.len() - 1);

                    let previous = self.wire_map.insert(name, id);
                    debug_assert_eq!(
                        previous, None,
                        "wire names are unique because they're picked from the map above"
                    );
                }
                Connector::Bus { range } => {
                    let previous = self.bus_map.insert(
                        name,
                        BuilderBus {
                            range,
                            wires: HashMap::new(),
                            kind: BuilderBusKind::Explicit,
                        },
                    );

                    debug_assert_eq!(
                        previous, None,
                        "bus names are unique because they're picked from the map above"
                    );
                }
            }
        }

        for (name, io) in ios {
            match io {
                GenericIo::Input(input) => self.add_input(name, input)?,
                GenericIo::Output(output) => self.add_output(name, output)?,
            }
        }

        for (name, component) in components {
            self.add_generic_component(&name, component)?;
        }

        for ((name, index), signal) in assignments {
            self.add_assignment(
                &name,
                match index {
                    Some(index) => Assignment::Bus { index, signal },
                    None => Assignment::Wire(signal),
                },
            )?;
        }

        Ok(Simulation {
            wires: self.wires,
            components: self.components,
            wire_updates: HashSet::new(),
            component_updates: HashSet::new(),
            inputs: self.inputs,
            outputs: self.outputs,
            fault: None,
            maybe_fault_targets: Vec::new(),
        })
    }
}
