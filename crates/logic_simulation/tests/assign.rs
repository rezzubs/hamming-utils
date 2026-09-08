//! Tests for wire assignments.

use logic_simulation::{
    Signal, Simulation,
    builder::{Assignment, BuildError, buffer, input, output, output_bus},
};

#[test]
fn wire_assignment() {
    let mut simulation = Simulation::<()>::builder()
        .with_io(output("out"))
        .with_assignment("out", Assignment::Wire(Signal::High))
        .build()
        .unwrap();

    simulation.settle();
    assert_eq!(simulation.read_wire("out").unwrap(), Signal::High);
}

#[test]
fn assignment_conflict() {
    let mut simulation = Simulation::<()>::builder()
        .with_io_many([input("in"), output("out")])
        .with_component("buf", buffer("in", "out"))
        .with_assignment("out", Assignment::Wire(Signal::High))
        .build()
        .unwrap();

    simulation.write_wire("in", Signal::High).unwrap();
    simulation.settle();
    assert_eq!(simulation.read_wire("out").unwrap(), Signal::High);

    // The assignment is driving high and the input low so there should be a conflict.
    simulation.write_wire("in", Signal::Low).unwrap();
    simulation.settle();
    assert_eq!(simulation.read_wire("out").unwrap(), Signal::Unknown);
}

#[test]
fn bus_assignment() {
    let mut simulation = Simulation::<()>::builder()
        .with_io(output_bus("out", 0..2))
        .with_assignment(
            "out",
            Assignment::Bus {
                index: 0,
                signal: Signal::High,
            },
        )
        .with_assignment(
            "out",
            Assignment::Bus {
                index: 1,
                signal: Signal::Low,
            },
        )
        .build()
        .unwrap();

    simulation.settle();

    let bus = simulation.output_bus("out").unwrap();
    assert_eq!(
        bus.read_vector(0..2).unwrap(),
        [Some(Signal::High), Some(Signal::Low)]
    );

    simulation.settle();
}

#[test]
fn assignment_undefined() {
    let result = Simulation::<()>::builder()
        .with_io(output_bus("out", 0..1))
        // `out` is a bus but we're assigning to a regular wire
        .with_assignment("out", Assignment::Wire(Signal::High))
        .build();

    assert_eq!(
        result,
        Err(BuildError::AssignToUndefinedWire { name: "out".into() })
    );

    let result = Simulation::<()>::builder()
        // `out` is not defined.
        .with_assignment("out", Assignment::Wire(Signal::High))
        .build();

    assert_eq!(
        result,
        Err(BuildError::AssignToUndefinedWire { name: "out".into() })
    );

    let result = Simulation::<()>::builder()
        .with_io(output("out"))
        // `out` is a wire but we're assigning to a bus
        .with_assignment(
            "out",
            Assignment::Bus {
                index: 0,
                signal: Signal::High,
            },
        )
        .build();

    assert_eq!(
        result,
        Err(BuildError::AssignToUndefinedBus { name: "out".into() })
    );

    let result = Simulation::<()>::builder()
        // `out` is not defined.
        .with_assignment("out", Assignment::Wire(Signal::High))
        .build();

    assert_eq!(
        result,
        Err(BuildError::AssignToUndefinedWire { name: "out".into() })
    );

    let result = Simulation::<()>::builder()
        .with_io_fallible(output("out").map(("out", 1)))
        .unwrap()
        // `out` is defined but index 0 is not mapped.
        .with_assignment(
            "out",
            Assignment::Bus {
                index: 0,
                signal: Signal::High,
            },
        )
        .build();

    assert_eq!(
        result,
        Err(BuildError::AssignToUndefinedBusIndex {
            name: "out".into(),
            index: 0
        })
    );
}
