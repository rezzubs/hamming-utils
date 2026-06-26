//! The [`Simulation`] type and related components.

pub mod builder;
pub mod bus;
mod io;
mod wire;

use crate::{
    Signal,
    components::{Component, GenericComponent},
    fault::Fault,
};
use builder::SimulationBuilder;
use bus::{InputBusView, OutputBusView};
use io::{Input, Output};
use std::collections::{HashMap, HashSet};

pub(crate) use wire::{OutputLink, Wire};

/// Construct a closure that triggers a wire update on the given `Simulation`.
/// This cannot be created as a function because a function would claim
/// exclusive ownership of `$self`.
macro_rules! trigger_wire_update {
    ($self:expr) => {
        |wire_id| {
            // we don't care if there is an existing value.
            _ = $self.wire_updates.insert(wire_id)
        }
    };
}
pub(crate) use trigger_wire_update;

/// Error cases for reading signals to wires.
///
/// See [`Simulation::read_wire`]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WireReadError {
    /// There is no connection with the given name.
    #[error("there is no input with name `{0}`")]
    NameUndefined(String),
    /// The requested connection is a bus, not a wire.
    #[error("the input `{0}` is a bus, not a wire")]
    Bus(String),
}

/// Error cases for writing signals to wires.
///
/// [`Simulation::write_wire`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WireWriteError {
    /// There is no connection with the given name.
    #[error("there is no input with name `{0}`")]
    NameUndefined(String),
    /// The requested connection is a bus, not a wire.
    #[error("the input `{0}` is a bus, not a wire")]
    Bus(String),
}

/// Error cases for accessing an input bus.
///
/// See [`Simulation::input_bus`]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InputBusLocateError {
    /// There is no connection with the given name.
    #[error("there is no input with name `{0}`")]
    NameUndefined(String),
    /// There is a connection with the given name, but it is a wire, not a bus.
    #[error("the input `{0}` is a wire, not a bus")]
    Wire(String),
}

/// Error cases for accessing an output bus.
///
/// See [`Simulation::output_bus`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum OutputBusLocateError {
    /// There is no connection with the given name.
    #[error("there is no output with name `{0}`")]
    NameUndefined(String),
    /// There is a connection with the given name, but it is a wire, not a bus.
    #[error("the output `{0}` is a wire, not a bus")]
    Wire(String),
}

/// A unique identifier for a wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct WireId(pub usize);

/// A unique identifier for a component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ComponentId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TargettedFault<F> {
    target: usize,
    fault: F,
}

/// An error for [`Simulation::make_faulty`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
#[error("the fault {0} is out of bounds")]
pub struct FaultOutOfBoundsError(pub usize);

/// A digital logic simulation.
///
/// See [`Simulation::builder`] for details on constructing a simulation.
///
/// At a high level the simulation is a collection of primitive components which
/// can be connected in a graph. The simulation tracks connections between
/// components and propagates updates. Use [`Simulation::settle`] to run updates
/// until the state stabilizes.
///
/// A wire may be connected to an arbitrary number of both input and output
/// ports in various components. Wires in the simulation can either be
/// standalone or an ordered collection called a [`Bus`](bus::Bus). The same API
/// is exposed as IO for the caller. A wire/bus can optionally be configured as an
/// input, output, or both. Configuring a wire as such enables values to be read
/// from or written to it.
///
/// After construction, the simulation is a black box. The caller can write
/// [`Signal`]s to input ports and read [`Signal`]s from output ports. There are
/// 4 methods exposed for IO:
/// - [`read_wire`](Simulation::read_wire)
/// - [`write_wire`](Simulation::write_wire)
/// - [`input_bus`](Simulation::input_bus)
/// - [`output_bus`](Simulation::output_bus)
///
/// The `*bus` methods return a `view` object that's used for writing/reading
/// scalar or vector signals to/from a bus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Simulation<F = ()> {
    wires: Vec<Wire>,
    /// Components are grouped by their number of outputs.
    components: Vec<GenericComponent>,

    wire_updates: HashSet<WireId>,
    component_updates: HashSet<ComponentId>,

    inputs: HashMap<String, Input>,
    outputs: HashMap<String, Output>,

    /// The component that is faulty if any.
    fault: Option<TargettedFault<F>>,
    /// Only initialized when a fault is set for the first time. We use an empty
    /// Vec to detect that it hasn't been initialized yet. Use [`Self::fault_targets`]
    /// to guarantee initialization.
    maybe_fault_targets: Vec<OutputLink>,
}

impl<F> Simulation<F> {
    /// Construct a new simulation builder.
    ///
    /// The builder has two kinds of methods for adding elements to the
    /// simulation. `add_*` for `&mut self` reference and `with_*` for an owned
    /// `self`. The documentation may refer to one of these but both exist for
    /// all cases.
    ///
    /// The builder has namespaces for three types of elements:
    /// - Connectors - a wire or a bus
    /// - Components - a primitive component.
    /// - IOs - input/output ports for the whole simulation.
    ///
    /// Out of these, only the IO namespace exists after construction. The rest
    /// are only used to resolve the actual connections during construction.
    ///
    /// Adding components and IOs can be a bit verbose so the [`builder`] module
    /// defines various of helper functions for common configurations.
    ///
    /// # Connectors
    ///
    /// At the lowest level, all connections are represented as wires.
    /// Components and IOs can read and write to these wires. If multiple
    /// elements write to a wire the signal is resovled with [`Signal::join`].
    ///
    /// A bus is an named and ordered collection of wires. Busses don't actually
    /// exist inside a simulation; they exist only as notational convenience and
    /// IO access. An indexed wire in a bus is referred to as a **bit**.
    ///
    /// Connections from a component or an IO port can be expressed using the
    /// [`Connection`](builder::Connection) type.
    ///
    /// A connector can be added manually using the
    /// [`add_wire`](SimulationBuilder::add_wire) and
    /// [`add_bus`](SimulationBuilder::add_bus) methods. Doing so is generally
    /// not necessary as using the name of an undefined connector will
    /// implicitly define it. Adding a bus explicitly will make it so that all
    /// connections to that bus are checked for correctness. For implicit bus
    /// connectors, the range is defined as the `min..max` of all encountered
    /// bus indices for that name.
    ///
    /// # Components
    ///
    /// These represent the foundational primitive components of the simulation.
    /// The components are defined by a
    /// [`ComponentSpec`](builder::ComponentSpec). Each component declares the
    /// connections of its ports. A `port` in the
    /// [`spec`](builder::ComponentSpec) is an optional connection to a
    /// connector.
    ///
    /// # IOs
    ///
    /// These declare the API of the final simulation that the caller will
    /// interact with after creation. An IO is either an input or output but
    /// both have the same interface - A wire or a bus. An IO port is like a
    /// connector but instead of connecting components it connects internal
    /// connectors to the outside world. An IO wire or a bus bit can be
    /// connected to any single internal connector in the simulation. A bit in a
    /// bus IO can connect to a number of internal wires or bus bits. For
    /// example it's possible to have an IO bus like this:
    ///
    /// ```text
    /// input bus "a":
    ///   4: "a"
    ///   5: nothing
    ///   6: "b"[7]
    /// ```
    ///
    /// Or visually like this (internal connectors left-to-right, IO top-to-bottom):
    ///
    /// ```text
    ///      a[4] a[5] a[6]
    ///        |         |
    /// b[6]---|---------|----
    /// b[7]---|---------*----
    /// b[8]---|--------------
    ///        |
    /// a   ---*--------------
    ///
    /// In this example, `a` is the name of the input bus. The bus has a range
    /// of `4..7`. Bit 4 is connected to wire `a`; remember that the wires
    /// (connectors) have a different namespace from IO. Bit 5 isn't connected to
    /// anything. Bit 6 is connected to bit 7 of bus `b`.
    ///
    /// As a reminder, there are wires and busses which are internal to the
    /// simulation and then there are IO wires and busses which are used as
    /// views to the internal connections. An IO bus can connect to many
    /// different internal connectors. This level of flexibility is not needed
    /// for most cases. The most common bus input will map all of its indices
    /// 1-1 to an internal bus with the same name - effectively making the
    /// internal bus public.
    ///
    /// For this use case the [`builder`] module provides functions
    /// [`input_bus`](builder::input_bus) and
    /// [`output_bus`](builder::output_bus). These return a builder object which
    /// implements `Into<NamedIo>` so it can be used with
    /// [`SimulationBuilder::add_io`]. The bus can then be further customized
    /// with the [`map`](builder::IoBusBuilder::map) method but doing this makes
    /// the builder fallible and must be used with
    /// [`SimulationBuilder::add_io_fallible`].
    ///
    /// Scalar IOs can be constructed with [`input`](builder::input) and
    /// [`output`](builder::output).
    ///
    /// # Assignments
    ///
    /// It is possible to assign a value to a connector using
    /// [`SimulationBuilder::add_assignment`] or
    /// [`SimulationBuilder::with_assignment`]. This creates a permanent driver
    /// for the connector.
    ///
    /// Note that this does not implicitly define the connector. If not other
    /// components reference that name or if the connector isn't explicitly
    /// defined, the build function will error.
    pub fn builder() -> SimulationBuilder<F> {
        SimulationBuilder::new()
    }

    /// Writes a signal to a named input wire.
    pub fn write_wire(&mut self, input_name: &str, value: Signal) -> Result<(), WireWriteError> {
        let Some(input) = self.inputs.get(input_name) else {
            return Err(WireWriteError::NameUndefined(input_name.to_owned()));
        };

        let component = match input {
            Input::Wire(component_id) => &mut self.components[component_id.0],
            Input::Bus(_) => return Err(WireWriteError::Bus(input_name.to_owned())),
        };

        let output_port = match component {
            GenericComponent::Input(output_port) => output_port,
            other => unreachable!(
                "`inputs` should not store non-input components, got {:?}",
                other
            ),
        };

        output_port.write(value, trigger_wire_update!(self));

        Ok(())
    }

    /// Access an input bus by name.
    ///
    /// See [`Simulation`] for details.
    pub fn input_bus<'a>(
        &'a mut self,
        bus_name: &'a str,
    ) -> Result<InputBusView<'a>, InputBusLocateError> {
        let Some(input) = self.inputs.get(bus_name) else {
            return Err(InputBusLocateError::NameUndefined(bus_name.to_owned()));
        };

        let bus = match input {
            Input::Wire(_) => return Err(InputBusLocateError::Wire(bus_name.to_owned())),
            Input::Bus(bus) => bus,
        };

        Ok(InputBusView::new(
            bus,
            bus_name,
            &mut self.components,
            &mut self.wire_updates,
        ))
    }

    /// Read a signal from a named output wire.
    pub fn read_wire(&self, output_name: &str) -> Result<Signal, WireReadError> {
        let Some(input) = self.outputs.get(output_name) else {
            return Err(WireReadError::NameUndefined(output_name.to_owned()));
        };

        let wire = match input {
            Output::Wire(wire_id) => &self.wires[wire_id.0],
            Output::Bus(_) => return Err(WireReadError::Bus(output_name.to_owned())),
        };

        Ok(wire.value())
    }

    /// Access an output bus by name.
    ///
    /// See [`Simulation`] for details.
    pub fn output_bus<'a>(
        &'a self,
        bus_name: &'a str,
    ) -> Result<OutputBusView<'a>, OutputBusLocateError> {
        let Some(output) = self.outputs.get(bus_name) else {
            return Err(OutputBusLocateError::NameUndefined(bus_name.to_owned()));
        };

        let bus = match output {
            Output::Wire(_) => return Err(OutputBusLocateError::Wire(bus_name.to_owned())),
            Output::Bus(bus) => bus,
        };

        Ok(OutputBusView::new(bus, bus_name, &self.wires))
    }
}

impl<F> Simulation<F>
where
    F: Fault,
{
    /// Applies a fault to the simulation.
    ///
    /// `target` needs to be smaller than [`Self::fault_radix`].
    ///
    /// Overwrites a fault if one already exists.
    pub fn make_faulty(&mut self, target: usize, fault: F) -> Result<(), FaultOutOfBoundsError> {
        if target >= self.fault_radix() {
            return Err(FaultOutOfBoundsError(target));
        }

        let previous_target = self.fault.take().map(|fault| fault.target);
        self.fault = Some(TargettedFault { target, fault });

        // Schedule an update for the current and previous targets.
        for target in [Some(target), previous_target] {
            let Some(target) = target else {
                continue;
            };

            let target_link = self.fault_targets()[target];

            let output = self.components[target_link.component_id.0]
                .nth_output(target_link.output_index)
                .expect("invalid fault target");

            let Some(wire_id) = output.wire_id else {
                return Ok(());
            };

            self.wire_updates.insert(wire_id);
        }

        Ok(())
    }

    /// Remove a fault.
    pub fn remove_fault(&mut self) {
        self.fault = None;
    }

    /// Returns the total number of possible faults.
    pub fn fault_radix(&mut self) -> usize {
        self.fault_targets().len()
    }

    /// Runs the simulation until all wires and components have settled.
    pub fn settle(&mut self) {
        while !self.wire_updates.is_empty() || !self.component_updates.is_empty() {
            self.settle_wire_updates();
            self.settle_component_updates();
        }
    }

    fn fault_targets(&mut self) -> &[OutputLink] {
        if self.maybe_fault_targets.is_empty() {
            let mut targets: Vec<OutputLink> = self
                .wires
                .iter()
                .flat_map(|wire| wire.linked_outputs())
                .filter(|link| {
                    // Inputs are not valid fault targets even though we treat
                    // them as components otherwise.
                    !matches!(
                        self.components[link.component_id.0],
                        GenericComponent::Input(_)
                    )
                })
                .copied()
                .collect();

            // The order must be stable across different instantiations of the
            // same simulation structure. Would use a hashset and no dedup
            // otherwise.
            targets.sort_by_key(|link| (link.component_id.0, link.output_index));
            targets.dedup();
            self.maybe_fault_targets = targets;
        }

        &self.maybe_fault_targets
    }

    /// Update all wires which are currently queued.
    fn settle_wire_updates(&mut self) {
        if self.fault.is_some() {
            // Ensure the fault targets cache is populated before we split borrows,
            // since fault_targets() requires &mut self.
            let _ = self.fault_targets();
        }

        let read_component_output = |output: OutputLink| {
            let component = &self.components[output.component_id.0];
            let signal = component
                .nth_output(output.output_index)
                .expect("Link's output index is out of bounds")
                .value;

            if let Some(fault) = &self.fault {
                let target = self.maybe_fault_targets[fault.target];
                if output == target {
                    return fault.fault.make_faulty(signal);
                }
            }

            signal
        };

        let mut trigger_component_update = |component_id| {
            _ = self.component_updates.insert(component_id);
        };

        for wire_id in self.wire_updates.drain() {
            self.wires[wire_id.0].update(read_component_output, &mut trigger_component_update);
        }
    }

    /// Update all components which are currently queued.
    fn settle_component_updates(&mut self) {
        let read_wire = |wire_id: WireId| self.wires[wire_id.0].value();

        for component_id in self.component_updates.drain() {
            self.components[component_id.0].update(read_wire, trigger_wire_update!(self));
        }
    }
}
