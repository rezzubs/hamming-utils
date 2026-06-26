//! Functions for creating a simulation from a verilog netlist.
//!
//! Right now we support a subset of NanGate cell types.

mod parsers;
mod port_map;

use crate::{
    Simulation,
    builder::{Assignment, BuildError, ElementKind, GateKind},
};
use ariadne::{Color, Label, ReportBuilder, ReportKind};
use chumsky::prelude::*;
use port_map::PortMap;
use std::{collections::HashMap, ops::Range};

pub use parsers::{Port, Statement, StatementKind, parser};

type ReportSpan = (String, Range<usize>);
/// An error report.
pub type Report<'src> = ariadne::Report<'src, ReportSpan>;

#[inline]
fn error_report<'src>(
    file_name: impl Into<String>,
    span: SimpleSpan,
) -> ReportBuilder<'src, ReportSpan> {
    Report::build(ReportKind::Error, (file_name.into(), span.into_range()))
        .with_config(ariadne::Config::new().with_index_type(ariadne::IndexType::Byte))
}

/// Parse statements from the netlist source.
pub fn parse_statements<'src>(
    src: &'src str,
    file_name: impl Into<String>,
) -> Result<Vec<Statement<'src>>, Vec<Report<'src>>> {
    let (statements, errors) = parser().parse(src).into_output_errors();
    let file_name = file_name.into();

    let errors = errors.into_iter().map(|err| {
        error_report(file_name.clone(), *err.span())
            .with_message(err.to_string())
            .with_label(
                Label::new((file_name.clone(), err.span().into_range()))
                    .with_message(err.reason().to_string())
                    .with_color(Color::Red),
            )
            .with_labels(err.contexts().map(|(label, span)| {
                Label::new((file_name.clone(), span.into_range()))
                    .with_message(format!("While parsing this {}", label))
                    .with_color(Color::Yellow)
            }))
            .finish()
    });

    statements.ok_or_else(|| errors.collect())
}

// no point in allocating and immediately dereferencing
#[allow(clippy::result_large_err)]
fn realize_port_map<'src>(
    file_name: impl Into<String>,
    ports: &[Port<'src>],
) -> Result<HashMap<String, Port<'src>>, Report<'src>> {
    // TODO: Check for the existance of wires.
    let file_name = file_name.into();

    let mut map = HashMap::new();
    for port in ports {
        let previous = map.insert(port.name.to_string(), port.clone());
        if let Some(previous) = previous {
            return Err(error_report(file_name.clone(), port.span)
                .with_message(format!("Port {} is defined twice", port.name))
                .with_label(
                    Label::new((file_name.clone(), previous.span.into_range()))
                        .with_message("First definition here")
                        .with_color(Color::Yellow),
                )
                .with_label(
                    Label::new((file_name.clone(), port.span.into_range()))
                        .with_message("Second definition here")
                        .with_color(Color::Red),
                )
                .finish());
        }
    }

    Ok(map)
}

fn error_label(
    file_name: impl Into<String>,
    span: SimpleSpan,
    message: impl Into<String>,
) -> Label<ReportSpan> {
    Label::new((file_name.into(), span.into_range()))
        .with_message(message.into())
        .with_color(Color::Red)
}

fn warn_label(
    file_name: impl Into<String>,
    span: SimpleSpan,
    message: impl Into<String>,
) -> Label<ReportSpan> {
    Label::new((file_name.into(), span.into_range()))
        .with_message(message.into())
        .with_color(Color::Yellow)
}

/// Create a [`Simulation`] from a list of [`Statement`]s.
pub fn simulation_from_statements<'src, F>(
    statements: &[Statement<'src>],
    file_name: impl Into<String>,
) -> Result<Simulation<F>, Vec<Report<'src>>> {
    let file_name = file_name.into();
    let mut simulation = Simulation::builder();

    let mut io_spans: HashMap<String, SimpleSpan> = HashMap::new();
    let mut cell_spans: HashMap<String, SimpleSpan> = HashMap::new();
    let mut assignment_spans: HashMap<(String, Option<usize>), SimpleSpan> = HashMap::new();

    let mut errors: Vec<Report<'src>> = Vec::new();

    use crate::builder::{input, input_bus, output, output_bus};

    let mut check_io_collision = |statement: &Statement| {
        let previous = io_spans.insert(statement.name.to_owned(), statement.span);

        previous.map(|previous| {
            error_report(file_name.clone(), statement.span)
                .with_message("IO name already in use")
                .with_label(error_label(&file_name, previous, "First use here"))
                .with_label(warn_label(
                    &file_name,
                    statement.span,
                    "Second definition here",
                ))
                .finish()
        })
    };

    for statement in statements {
        match &statement.kind {
            StatementKind::Input => {
                if let Some(report) = check_io_collision(statement) {
                    errors.push(report);
                }
                simulation.add_io(input(statement.name));
            }
            StatementKind::InputBus { range } => {
                if let Some(report) = check_io_collision(statement) {
                    errors.push(report);
                }
                simulation.add_io(input_bus(statement.name, range.clone()));
            }
            StatementKind::Output => {
                if let Some(report) = check_io_collision(statement) {
                    errors.push(report);
                }
                simulation.add_io(output(statement.name));
            }
            StatementKind::OutputBus { range } => {
                if let Some(report) = check_io_collision(statement) {
                    errors.push(report);
                }
                simulation.add_io(output_bus(statement.name, range.clone()));
            }
            StatementKind::Wire => {
                simulation.add_wire(statement.name);
            }
            StatementKind::Bus { range } => {
                simulation.add_bus(statement.name, range.clone());
            }
            StatementKind::Assignment { index, value } => {
                let name = statement.name;
                let span = statement.span;
                if let Some(previous) = assignment_spans.insert((name.to_owned(), *index), span) {
                    errors.push(
                        error_report(file_name.clone(), span)
                            .with_message(match index {
                                Some(index) => {
                                    format!("{}[{}] was assigned to more than once.", name, index)
                                }
                                None => format!("{} was assigned to more than once.", name),
                            })
                            .with_label(error_label(
                                &file_name,
                                statement.span,
                                "Second assignment here",
                            ))
                            .with_label(warn_label(&file_name, previous, "First assignment here"))
                            .finish(),
                    );
                }

                simulation.add_assignment(
                    name,
                    match index {
                        Some(index) => Assignment::Bus {
                            index: *index,
                            signal: *value,
                        },
                        None => Assignment::Wire(*value),
                    },
                );
            }
            StatementKind::Cell {
                name: cell_name,
                name_span,
                ports_span,
                ports,
            } => {
                let previous = cell_spans.insert(statement.name.to_owned(), statement.span);

                previous.map(|previous| {
                    error_report(file_name.clone(), statement.span)
                        .with_message(format!("Duplicate cell name `{}`", statement.name))
                        .with_label(error_label(
                            &file_name,
                            statement.span,
                            "Second definition here",
                        ))
                        .with_label(warn_label(&file_name, previous, "First definition here"))
                        .finish()
                });

                let ports = match realize_port_map(file_name.clone(), ports) {
                    Ok(ports) => ports,
                    Err(err) => {
                        errors.push(err);
                        continue;
                    }
                };

                let ports = PortMap {
                    map: ports,
                    cell_name,
                    file_name: file_name.clone(),
                    ports_span: *ports_span,
                };

                let component = match *cell_name {
                    "AND2" => ports.build_gate2(GateKind::And),
                    "OR2" => ports.build_gate2(GateKind::Or),
                    "NAND2" => ports.build_gate2(GateKind::Nand),
                    "NOR2" => ports.build_gate2(GateKind::Nor),

                    "AND3" => ports.build_gate3(GateKind::And),
                    "OR3" => ports.build_gate3(GateKind::Or),
                    "NAND3" => ports.build_gate3(GateKind::Nand),
                    "NOR3" => ports.build_gate3(GateKind::Nor),

                    "AND4" => ports.build_gate4(GateKind::And),
                    "OR4" => ports.build_gate4(GateKind::Or),
                    "NAND4" => ports.build_gate4(GateKind::Nand),
                    "NOR4" => ports.build_gate4(GateKind::Nor),

                    "XOR2" => ports.build_xor2(),
                    "XNOR2" => ports.build_xnor2(),

                    "CLKBUF" => ports.build_buffer(),
                    "INV" => ports.build_inverter(),
                    "DLH" => ports.build_dlatch(),
                    "MUX2" => ports.build_mux2(),

                    "AOI21" => ports.build_aoi21(),
                    "AOI22" => ports.build_aoi22(),
                    "AOI211" => ports.build_aoi211(),
                    "AOI221" => ports.build_aoi221(),
                    "AOI222" => ports.build_aoi222(),

                    "OAI21" => ports.build_oai21(),
                    "OAI22" => ports.build_oai22(),
                    "OAI211" => ports.build_oai211(),
                    "OAI221" => ports.build_oai221(),
                    "OAI222" => ports.build_oai222(),
                    "OAI33" => ports.build_oai33(),

                    "HA" => ports.build_half_adder(),
                    "FA" => ports.build_full_adder(),

                    _ => {
                        let accepted_kinds = [
                            "AND2", "AND3", "AND4", "OR2", "OR3", "OR4", "NAND2", "NAND3", "NAND4",
                            "NOR2", "NOR3", "NOR4", "XOR2", "XNOR2", "CLKBUF", "INV", "DLH",
                            "MUX2", "AOI21", "AOI22", "AOI211", "AOI221", "AOI222", "OAI21",
                            "OAI22", "OAI211", "OAI221", "OAI222", "OAI33", "HA", "FA",
                        ];
                        let report = error_report(file_name.clone(), *name_span)
                            .with_message(format!("Unknown cell {}", cell_name))
                            .with_label(
                                Label::new((file_name.clone(), name_span.into_range()))
                                    .with_message("Unknown cell")
                                    .with_color(Color::Red),
                            )
                            .with_note(format!("Accepted kinds: {}", accepted_kinds.join(", ")))
                            .finish();

                        Err(report)
                    }
                };

                let component = match component {
                    Ok(component) => component,
                    Err(report) => {
                        errors.push(report);
                        continue;
                    }
                };

                simulation.add_component(statement.name, component);
            }
        }
    }

    let simulation = match simulation.build() {
        Ok(simulation) => simulation,
        Err(err) => {
            match &err {
                BuildError::BusIndexOutOfBounds {
                    element_name,
                    element_kind,
                    bus_name,
                    ..
                } => {
                    let element_span = match element_kind {
                        ElementKind::Input | ElementKind::Output => *io_spans
                            .get(element_name)
                            .expect("the element must exist if this error is generated"),
                        ElementKind::Component => *cell_spans
                            .get(element_name)
                            .expect("the element must exist if this error is generated"),
                    };
                    let bus_span = io_spans
                        .get(bus_name)
                        .expect("the bus must exist if this error is generated");

                    let report = error_report(file_name.clone(), element_span)
                        .with_message(err.to_string())
                        .with_labels([
                            warn_label(&file_name, *bus_span, "Bus defined here"),
                            error_label(&file_name, element_span, "Invalid connection here"),
                        ])
                        .finish();

                    errors.push(report);
                }
                BuildError::AssignToUndefinedWire { name } => {
                    let span = assignment_spans
                        .get(&(name.clone(), None))
                        .copied()
                        .expect("We should have tracked all spans");

                    errors.push(
                        error_report(file_name.clone(), span)
                            .with_message(err.to_string())
                            .with_label(error_label(&file_name, span, "Inalid assignment"))
                            .finish(),
                    );
                }
                BuildError::AssignToUndefinedBus { name } => {
                    let span = assignment_spans
                        .get(&(name.clone(), None))
                        .copied()
                        .expect("We should have tracked all spans");

                    errors.push(
                        error_report(file_name.clone(), span)
                            .with_message(err.to_string())
                            .with_label(error_label(&file_name, span, "Inalid assignment"))
                            .finish(),
                    );
                }
                BuildError::AssignToUndefinedBusIndex { name, index } => {
                    let span = assignment_spans
                        .get(&(name.clone(), Some(*index)))
                        .copied()
                        .expect("We should have tracked all spans");

                    errors.push(
                        error_report(file_name.clone(), span)
                            .with_message(err.to_string())
                            .with_label(error_label(&file_name, span, "Inalid assignment"))
                            .finish(),
                    );
                }
            }

            return Err(errors);
        }
    };

    if errors.is_empty() {
        Ok(simulation)
    } else {
        Err(errors)
    }
}

/// Parse a simulation from a string source.
pub fn parse_simulation<'src, F>(
    src: &'src str,
    file_name: impl Into<String>,
) -> Result<Simulation<F>, Vec<Report<'src>>> {
    let file_name = file_name.into();
    let statements = parse_statements(src, file_name.clone())?;

    simulation_from_statements(&statements, file_name)
}
