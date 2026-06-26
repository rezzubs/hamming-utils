use super::{Port, Report, error_report};
use crate::builder::{ComponentSpec, Connection, GateKind};
use ariadne::{Color, Label};
use chumsky::span::SimpleSpan;
use std::collections::HashMap;

/// A map of port names to ports, used for resolving port connections for the
/// different cell types.
pub(crate) struct PortMap<'src> {
    pub map: HashMap<String, Port<'src>>,
    pub cell_name: &'src str,
    pub file_name: String,
    pub ports_span: SimpleSpan,
}

// we would be dereferencing these immediately anyway.
#[allow(clippy::result_large_err)]
impl<'src> PortMap<'src> {
    /// Take a port from the map by name, returning a `Connection` if found. A
    /// single name can only be taken once.
    pub fn take(&mut self, port_name: &str) -> Result<Connection, Report<'src>> {
        let Some(port) = self.map.remove(port_name) else {
            let report = error_report(self.file_name.clone(), self.ports_span)
                .with_message(format!(
                    "Cell {} expects port {}",
                    self.cell_name, port_name
                ))
                .with_label(
                    Label::new((self.file_name.clone(), self.ports_span.into_range()))
                        .with_message(format!("Expected .{} here", port_name))
                        .with_color(Color::Red),
                )
                .finish();

            return Err(report);
        };

        let connection = match port.target.index {
            Some(index) => Connection::Bus {
                name: port.target.name.to_owned(),
                index,
            },
            None => Connection::Wire {
                name: port.target.name.to_owned(),
            },
        };

        Ok(connection)
    }

    /// Mark that all ports should have been consumed. If there are any
    /// remaining ports, an error is returned.
    pub fn finish(self) -> Result<(), Report<'src>> {
        if !self.map.is_empty() {
            let message = if self.map.len() == 1 {
                format!(
                    "Unexpected port: {}",
                    self.map.keys().next().expect("We checked the length")
                )
            } else {
                format!(
                    "Unexpected ports: {}",
                    self.map.into_keys().collect::<Vec<_>>().join(", ")
                )
            };

            return Err(error_report(&self.file_name, self.ports_span)
                .with_message(message)
                .with_label(
                    Label::new((self.file_name.clone(), self.ports_span.into_range()))
                        .with_message("Redundant ports here")
                        .with_color(Color::Red),
                )
                .finish());
        }

        Ok(())
    }

    pub fn build_gate2(mut self, kind: GateKind) -> Result<ComponentSpec, Report<'src>> {
        let component = ComponentSpec::Gate2 {
            kind,
            inputs: [Some(self.take("A1")?), Some(self.take("A2")?)],
            output: Some(self.take("ZN")?),
        };

        self.finish()?;

        Ok(component)
    }

    pub fn build_gate3(mut self, kind: GateKind) -> Result<ComponentSpec, Report<'src>> {
        let component = ComponentSpec::Gate3 {
            kind,
            inputs: [
                Some(self.take("A1")?),
                Some(self.take("A2")?),
                Some(self.take("A3")?),
            ],
            output: Some(self.take("ZN")?),
        };
        self.finish()?;
        Ok(component)
    }

    pub fn build_gate4(mut self, kind: GateKind) -> Result<ComponentSpec, Report<'src>> {
        let component = ComponentSpec::Gate4 {
            kind,
            inputs: [
                Some(self.take("A1")?),
                Some(self.take("A2")?),
                Some(self.take("A3")?),
                Some(self.take("A4")?),
            ],
            output: Some(self.take("ZN")?),
        };
        self.finish()?;
        Ok(component)
    }

    pub fn build_xor2(mut self) -> Result<ComponentSpec, Report<'src>> {
        let component = ComponentSpec::Gate2 {
            kind: GateKind::Xor,
            inputs: [Some(self.take("A")?), Some(self.take("B")?)],
            output: Some(self.take("Z")?),
        };
        self.finish()?;
        Ok(component)
    }

    pub fn build_xnor2(mut self) -> Result<ComponentSpec, Report<'src>> {
        let component = ComponentSpec::Gate2 {
            kind: GateKind::Xnor,
            inputs: [Some(self.take("A")?), Some(self.take("B")?)],
            output: Some(self.take("ZN")?),
        };
        self.finish()?;
        Ok(component)
    }

    pub fn build_buffer(mut self) -> Result<ComponentSpec, Report<'src>> {
        let component = ComponentSpec::Buffer {
            input: Some(self.take("A")?),
            output: Some(self.take("Z")?),
        };
        self.finish()?;
        Ok(component)
    }

    pub fn build_inverter(mut self) -> Result<ComponentSpec, Report<'src>> {
        let component = ComponentSpec::Inverter {
            input: Some(self.take("A")?),
            output: Some(self.take("ZN")?),
        };
        self.finish()?;
        Ok(component)
    }

    pub fn build_dlatch(mut self) -> Result<ComponentSpec, Report<'src>> {
        let component = ComponentSpec::DLatch {
            data: Some(self.take("D")?),
            enable: Some(self.take("G")?),
            output: Some(self.take("Q")?),
        };
        self.finish()?;
        Ok(component)
    }

    pub fn build_mux2(mut self) -> Result<ComponentSpec, Report<'src>> {
        let component = ComponentSpec::Mux2 {
            a: Some(self.take("A")?),
            b: Some(self.take("B")?),
            select: Some(self.take("S")?),
            output: Some(self.take("Z")?),
        };
        self.finish()?;
        Ok(component)
    }

    pub fn build_aoi21(mut self) -> Result<ComponentSpec, Report<'src>> {
        let component = ComponentSpec::AndOrInvert21 {
            a: Some(self.take("A")?),
            b1: Some(self.take("B1")?),
            b2: Some(self.take("B2")?),
            output: Some(self.take("ZN")?),
        };
        self.finish()?;
        Ok(component)
    }

    pub fn build_aoi22(mut self) -> Result<ComponentSpec, Report<'src>> {
        let component = ComponentSpec::AndOrInvert22 {
            a1: Some(self.take("A1")?),
            a2: Some(self.take("A2")?),
            b1: Some(self.take("B1")?),
            b2: Some(self.take("B2")?),
            output: Some(self.take("ZN")?),
        };
        self.finish()?;
        Ok(component)
    }

    pub fn build_aoi211(mut self) -> Result<ComponentSpec, Report<'src>> {
        let component = ComponentSpec::AndOrInvert211 {
            a: Some(self.take("A")?),
            b: Some(self.take("B")?),
            c1: Some(self.take("C1")?),
            c2: Some(self.take("C2")?),
            output: Some(self.take("ZN")?),
        };
        self.finish()?;
        Ok(component)
    }

    pub fn build_aoi221(mut self) -> Result<ComponentSpec, Report<'src>> {
        let component = ComponentSpec::AndOrInvert221 {
            a: Some(self.take("A")?),
            b1: Some(self.take("B1")?),
            b2: Some(self.take("B2")?),
            c1: Some(self.take("C1")?),
            c2: Some(self.take("C2")?),
            output: Some(self.take("ZN")?),
        };
        self.finish()?;
        Ok(component)
    }

    pub fn build_aoi222(mut self) -> Result<ComponentSpec, Report<'src>> {
        let component = ComponentSpec::AndOrInvert222 {
            a1: Some(self.take("A1")?),
            a2: Some(self.take("A2")?),
            b1: Some(self.take("B1")?),
            b2: Some(self.take("B2")?),
            c1: Some(self.take("C1")?),
            c2: Some(self.take("C2")?),
            output: Some(self.take("ZN")?),
        };
        self.finish()?;
        Ok(component)
    }

    pub fn build_oai21(mut self) -> Result<ComponentSpec, Report<'src>> {
        let component = ComponentSpec::OrAndInvert21 {
            a: Some(self.take("A")?),
            b1: Some(self.take("B1")?),
            b2: Some(self.take("B2")?),
            output: Some(self.take("ZN")?),
        };
        self.finish()?;
        Ok(component)
    }

    pub fn build_oai22(mut self) -> Result<ComponentSpec, Report<'src>> {
        let component = ComponentSpec::OrAndInvert22 {
            a1: Some(self.take("A1")?),
            a2: Some(self.take("A2")?),
            b1: Some(self.take("B1")?),
            b2: Some(self.take("B2")?),
            output: Some(self.take("ZN")?),
        };
        self.finish()?;
        Ok(component)
    }

    pub fn build_oai211(mut self) -> Result<ComponentSpec, Report<'src>> {
        let component = ComponentSpec::OrAndInvert211 {
            a: Some(self.take("A")?),
            b: Some(self.take("B")?),
            c1: Some(self.take("C1")?),
            c2: Some(self.take("C2")?),
            output: Some(self.take("ZN")?),
        };
        self.finish()?;
        Ok(component)
    }

    pub fn build_oai221(mut self) -> Result<ComponentSpec, Report<'src>> {
        let component = ComponentSpec::OrAndInvert221 {
            a: Some(self.take("A")?),
            b1: Some(self.take("B1")?),
            b2: Some(self.take("B2")?),
            c1: Some(self.take("C1")?),
            c2: Some(self.take("C2")?),
            output: Some(self.take("ZN")?),
        };
        self.finish()?;
        Ok(component)
    }

    pub fn build_oai222(mut self) -> Result<ComponentSpec, Report<'src>> {
        let component = ComponentSpec::OrAndInvert222 {
            a1: Some(self.take("A1")?),
            a2: Some(self.take("A2")?),
            b1: Some(self.take("B1")?),
            b2: Some(self.take("B2")?),
            c1: Some(self.take("C1")?),
            c2: Some(self.take("C2")?),
            output: Some(self.take("ZN")?),
        };
        self.finish()?;
        Ok(component)
    }

    pub fn build_oai33(mut self) -> Result<ComponentSpec, Report<'src>> {
        let component = ComponentSpec::OrAndInvert33 {
            a1: Some(self.take("A1")?),
            a2: Some(self.take("A2")?),
            a3: Some(self.take("A3")?),
            b1: Some(self.take("B1")?),
            b2: Some(self.take("B2")?),
            b3: Some(self.take("B3")?),
            output: Some(self.take("ZN")?),
        };
        self.finish()?;
        Ok(component)
    }

    pub fn build_half_adder(mut self) -> Result<ComponentSpec, Report<'src>> {
        let component = ComponentSpec::HalfAdder {
            a: Some(self.take("A")?),
            b: Some(self.take("B")?),
            sum: Some(self.take("S")?),
            carry: Some(self.take("CO")?),
        };
        self.finish()?;
        Ok(component)
    }

    pub fn build_full_adder(mut self) -> Result<ComponentSpec, Report<'src>> {
        let component = ComponentSpec::FullAdder {
            a: Some(self.take("A")?),
            b: Some(self.take("B")?),
            carry_in: Some(self.take("CI")?),
            output: Some(self.take("S")?),
            carry_out: Some(self.take("CO")?),
        };
        self.finish()?;
        Ok(component)
    }
}
