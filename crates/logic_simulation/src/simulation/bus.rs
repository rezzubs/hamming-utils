//! Bus connections on a [`Simulation`](crate::Simulation).
//!
//! See [`Simulation::input_bus`](crate::Simulation::input_bus) and
//! [`Simulation::output_bus`](crate::Simulation::output_bus) .

use std::{
    collections::{HashMap, HashSet},
    ops::Range,
};

use crate::{
    Signal,
    components::GenericComponent,
    simulation::{ComponentId, Wire, WireId, trigger_wire_update},
};

/// The error for [`Bus::new`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, thiserror::Error)]
#[error("index {invalid_index} is out of range {range:?}")]
pub struct BusCreationError {
    /// The range that was given.
    pub range: Range<usize>,
    /// The index that was out of `range`.
    pub invalid_index: usize,
}

/// The error for [`Bus::from_parts`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, thiserror::Error)]
#[error("range length {range_len} does not match items length {items_len}")]
pub struct FromPartsError {
    /// The length of the given range.
    pub range_len: usize,
    /// The length of the given items.
    pub items_len: usize,
}

/// A generic container for connections. Maps port a range of port indices to
/// wires. Ports can remain disconnected.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Bus<T> {
    /// The range the bus was configured with. This is used to translate indices
    /// for `items`.
    range: Range<usize>,
    /// These are the linked items. The Vec always has the same length as the
    /// range. Disconnected ports are represented by `None`.
    items: Vec<Option<T>>,
}

impl<T> Bus<T>
where
    T: Clone,
{
    /// Create a new `Bus` from a range and a map of port indices to IDs.
    ///
    /// Returns an error if any index is out of range.
    pub fn new(range: Range<usize>, items: &HashMap<usize, T>) -> Result<Self, BusCreationError> {
        let mut ids = vec![None; range.len()];
        for (&index, id) in items.iter() {
            if !range.contains(&index) {
                return Err(BusCreationError {
                    range,
                    invalid_index: index,
                });
            }
            ids[index - range.start] = Some(id.clone());
        }

        Ok(Self { range, items: ids })
    }

    /// Create a `Bus` from a range and a `Vec` of items.
    ///
    /// Returns an error if the range length does not match the items length.
    pub fn from_parts(range: Range<usize>, items: Vec<Option<T>>) -> Result<Self, FromPartsError> {
        if range.len() != items.len() {
            return Err(FromPartsError {
                range_len: range.len(),
                items_len: items.len(),
            });
        }
        Ok(Self { range, items })
    }

    /// Access the range of indices this bus is configured with.
    pub fn range(&self) -> &Range<usize> {
        &self.range
    }

    /// Access the items in this bus.
    pub fn items(&self) -> &[Option<T>] {
        &self.items
    }

    /// Consume the bus and access the internal range and items.
    pub fn into_inner(self) -> (Range<usize>, Vec<Option<T>>) {
        (self.range, self.items)
    }

    /// Consume the bus and access the internal range only.
    pub fn into_range(self) -> Range<usize> {
        self.range
    }

    /// Consume the bus and access the internal items only.
    pub fn into_items(self) -> Vec<Option<T>> {
        self.items
    }
}

/// An error that occurs when accessing a bus index that is out of bounds.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("the bus `{name}` has a range {range:?}, got index {index}")]
pub struct BusOutOfBoundsError {
    /// The name of the bus.
    pub name: String,
    /// The range of valid indices for the bus.
    pub range: Range<usize>,
    /// The index that was out of bounds.
    pub index: usize,
}

/// An error for reading/writing vector signals from/to a bus.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BusVectorMismatchError {
    /// The bus does not fully contain the given range.
    #[error("the bus has a range {range:?}, which doesn't contain the given range {indices:?}")]
    RangeMismatch {
        /// The range the bus is configured with.
        range: Range<usize>,
        /// The range of indices that was requested.
        indices: Range<usize>,
    },
    /// The given range does not match the length of the given values.
    #[error("the range length ({range_length}) doesn't match the values length ({values_length})")]
    InvalidRange {
        /// The length of the given range.
        range_length: usize,
        /// The length of the given values.
        values_length: usize,
    },
}

/// A type that's used to read signals from an input bus.
///
/// Construct with [`Simulation::input_bus`](crate::Simulation::input_bus).
pub struct InputBusView<'a> {
    /// The bus to read from.
    bus: &'a Bus<ComponentId>,
    /// The name of the bus.
    name: &'a str,
    /// A reference to the components of a [`crate::Simulation`].
    components: &'a mut Vec<GenericComponent>,
    /// A reference wire updates of a [`crate::Simulation`].
    wire_updates: &'a mut HashSet<WireId>,
}

impl<'a> InputBusView<'a> {
    /// Create a new input bus view.
    // NOTE: This function exists to keep the fields private.
    pub(crate) fn new(
        bus: &'a Bus<ComponentId>,
        name: &'a str,
        components: &'a mut Vec<GenericComponent>,
        wire_updates: &'a mut HashSet<WireId>,
    ) -> Self {
        Self {
            bus,
            name,
            components,
            wire_updates,
        }
    }

    /// Returns the range of indices this bus supports.
    pub fn range(&self) -> Range<usize> {
        self.bus.range.clone()
    }

    /// Write a scalar value into the bus using an index into the [`Bus::items`] array.
    ///
    /// If that port is not connected to anything then it's a no-op.
    ///
    /// # Panics
    ///
    /// Panics if `raw_index` is out of bounds.
    pub fn write_scalar_raw(&mut self, raw_index: usize, value: Signal) {
        let Some(component_id) = self.bus.items[raw_index] else {
            return;
        };
        let output_port = match &mut self.components[component_id.0] {
            GenericComponent::Input(output_port) => output_port,
            other => unreachable!(
                "`inputs` should not store non-input components, got {:?}",
                other
            ),
        };
        output_port.write(value, trigger_wire_update!(self));
    }

    /// Write a scalar signal to the bus using a mapped index.
    ///
    /// Returns an error if the configured range does not contain the index.
    pub fn write_scalar(&mut self, index: usize, value: Signal) -> Result<(), BusOutOfBoundsError> {
        if !self.bus.range.contains(&index) {
            return Err(BusOutOfBoundsError {
                name: self.name.to_owned(),
                range: self.bus.range.clone(),
                index,
            });
        }
        let raw_index = index - self.bus.range.start;

        self.write_scalar_raw(raw_index, value);

        Ok(())
    }

    /// Write a vector signal into the bus.
    ///
    /// # Invariants
    ///
    /// - `indices` must be a range that is contained within the bus's configured range.
    /// - `values` must have the same length as `indices`.
    pub fn write_vector(
        &mut self,
        indices: Range<usize>,
        values: &[Signal],
    ) -> Result<(), BusVectorMismatchError> {
        if values.len() != indices.len() {
            return Err(BusVectorMismatchError::InvalidRange {
                range_length: indices.len(),
                values_length: values.len(),
            });
        }

        if !(self.bus.range.contains(&indices.start)
            && self.bus.range.contains(&indices.end.saturating_sub(1)))
        {
            return Err(BusVectorMismatchError::RangeMismatch {
                range: self.bus.range.clone(),
                indices: indices.clone(),
            });
        }

        for (index, value) in values.iter().enumerate() {
            let raw_index = indices.start + index;

            self.write_scalar_raw(raw_index, *value);
        }

        Ok(())
    }
}

/// A type that's used to write signals to an output bus.
///
/// Construct with [`Simulation::output_bus`](crate::Simulation::output_bus).
pub struct OutputBusView<'a> {
    bus: &'a Bus<WireId>,
    name: &'a str,
    wires: &'a Vec<Wire>,
}

impl<'a> OutputBusView<'a> {
    /// Construct a new `OutputBusView`.
    // NOTE: This function exists to keep the fields private
    pub(crate) fn new(bus: &'a Bus<WireId>, name: &'a str, wires: &'a Vec<Wire>) -> Self {
        Self { bus, name, wires }
    }

    /// Returns the range of indices this bus supports.
    pub fn range(&self) -> Range<usize> {
        self.bus.range.clone()
    }

    /// Read a scalar value from the bus using an index into the [`Bus::items`] array.
    ///
    /// Returns `None` if this output is disconnected.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn read_scalar_raw(&self, raw_index: usize) -> Option<Signal> {
        let wire_id = self.bus.items[raw_index]?;
        Some(self.wires[wire_id.0].value())
    }

    /// Read a scalar value from the bus using a mapped index.
    ///
    /// Returns `None` if this output is disconnected. Returns an error if the
    /// configured range doesn't contain the index.
    pub fn read_scalar(&self, index: usize) -> Result<Option<Signal>, BusOutOfBoundsError> {
        if !self.bus.range.contains(&index) {
            return Err(BusOutOfBoundsError {
                name: self.name.to_owned(),
                range: self.bus.range.clone(),
                index,
            });
        }
        let raw_index = index - self.bus.range.start;

        Ok(self.read_scalar_raw(raw_index))
    }

    /// Read a vector signal from the bus using a range of mapped indices.
    ///
    /// Returns an error if the configured range doesn't contain the indices.
    pub fn read_vector(
        &self,
        indices: Range<usize>,
    ) -> Result<Vec<Option<Signal>>, BusVectorMismatchError> {
        if !(self.bus.range.contains(&indices.start)
            && self.bus.range.contains(&indices.end.saturating_sub(1)))
        {
            return Err(BusVectorMismatchError::RangeMismatch {
                range: self.bus.range.clone(),
                indices: indices.clone(),
            });
        }

        Ok(indices
            .map(|index| {
                let raw_index = index - self.bus.range.start;

                self.read_scalar_raw(raw_index)
            })
            .collect())
    }
}
