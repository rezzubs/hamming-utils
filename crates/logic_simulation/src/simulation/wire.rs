use std::collections::HashSet;

use super::ComponentId;
use crate::Signal;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct OutputLink {
    pub component_id: ComponentId,
    pub output_index: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct Wire {
    /// The current value of the wire.
    value: Signal,
    /// The wire will be added to the update queue whenever the values of any of
    /// these output ports change. The wire will update its value based on the
    /// values of these outputs at the time of the update.
    linked_outputs: HashSet<OutputLink>,
    /// The components which have input ports linked to this wire. These
    /// components will be notified when the wire updates
    linked_inputs: HashSet<ComponentId>,
}

impl Wire {
    /// Create a new wire with an unknown value and no linked components.
    pub fn new() -> Self {
        Self::default()
    }

    /// Link this wire with a component's output port.
    pub fn link_output(&mut self, output: OutputLink) {
        self.linked_outputs.insert(output);
    }

    /// Link this wire with a component's input port.
    pub fn link_input(&mut self, input: ComponentId) {
        self.linked_inputs.insert(input);
    }

    /// Returns the current value of the wire.
    #[doc(alias = "read")]
    pub fn value(&self) -> Signal {
        self.value
    }

    pub fn linked_outputs(&self) -> &HashSet<OutputLink> {
        &self.linked_outputs
    }

    /// Update the wire's value and trigger updates in connected components.
    ///
    /// # Panics
    ///
    /// Panics if `components` doesn't contain a connected component.
    pub fn update<R, T>(&mut self, read_component_output: R, mut trigger_component_update: T)
    where
        R: Fn(OutputLink) -> Signal,
        T: FnMut(ComponentId),
    {
        let mut new_value: Option<Signal> = None;
        for output in &self.linked_outputs {
            let output_signal = read_component_output(*output);

            let Some(new_value) = &mut new_value else {
                new_value = Some(output_signal);
                continue;
            };

            *new_value = new_value.join(output_signal);
        }

        let Some(new_value) = new_value else {
            return;
        };

        if self.value != new_value {
            self.value = new_value;
            for &linked_component in &self.linked_inputs {
                trigger_component_update(linked_component);
            }
        }
    }
}
