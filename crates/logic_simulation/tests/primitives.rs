use logic_simulation::{
    Signal, Simulation,
    builder::{
        and2, and4, aoi21, aoi22, aoi211, aoi221, aoi222, dlatch, full_adder, half_adder, high,
        input, inverter, low, mux2, nand2, oai21, oai22, oai33, oai211, oai221, oai222, or2,
        output, unknown, xnor2, xor2, xor4,
    },
};
use proptest::prelude::*;

fn check(simulation: &mut Simulation, inputs: impl IntoIterator<Item = Signal>, expected: Signal) {
    for (index, value) in inputs.into_iter().enumerate() {
        simulation.write_wire(&index.to_string(), value).unwrap();
    }
    simulation.settle();
    assert_eq!(simulation.read_wire("z").unwrap(), expected);
}

#[test]
fn test_and2() {
    let mut simulation = Simulation::builder()
        .with_component("and", and2(["0", "1"], "z"))
        .with_io_many([input("0"), input("1"), output("z")])
        .build()
        .unwrap();

    let truth_table = [
        ([low(), low()], low()),
        ([low(), high()], low()),
        ([low(), unknown()], low()),
        ([high(), low()], low()),
        ([high(), high()], high()),
        ([high(), unknown()], unknown()),
        ([unknown(), low()], low()),
        ([unknown(), high()], unknown()),
        ([unknown(), unknown()], unknown()),
    ];

    for (input, output) in truth_table {
        check(&mut simulation, input, output);
    }
}

#[test]
fn test_and4() {
    let mut simulation = Simulation::builder()
        .with_component("and", and4(["0", "1", "2", "3"], "z"))
        .with_io_many([input("0"), input("1"), input("2"), input("3"), output("z")])
        .build()
        .unwrap();

    let truth_table = [
        ([high(), high(), high(), high()], high()),
        ([low(), high(), high(), high()], low()),
        ([high(), low(), high(), high()], low()),
        ([high(), high(), low(), high()], low()),
        ([high(), high(), high(), low()], low()),
        ([low(), low(), low(), low()], low()),
    ];

    for (input, output) in truth_table {
        check(&mut simulation, input, output);
    }
}

#[test]
fn test_or2() {
    let mut simulation = Simulation::builder()
        .with_component("and", or2(["0", "1"], "z"))
        .with_io_many([input("0"), input("1"), output("z")])
        .build()
        .unwrap();

    let truth_table = [
        ([low(), low()], low()),
        ([low(), high()], high()),
        ([low(), unknown()], unknown()),
        ([high(), low()], high()),
        ([high(), high()], high()),
        ([high(), unknown()], high()),
        ([unknown(), low()], unknown()),
        ([unknown(), high()], high()),
        ([unknown(), unknown()], unknown()),
    ];

    for (input, output) in truth_table {
        check(&mut simulation, input, output);
    }
}

#[test]
fn test_nand2() {
    let mut simulation = Simulation::builder()
        .with_component("nand", nand2(["0", "1"], "z"))
        .with_io_many([input("0"), input("1"), output("z")])
        .build()
        .unwrap();

    let truth_table = [
        ([low(), low()], high()),
        ([low(), high()], high()),
        ([low(), unknown()], high()),
        ([high(), low()], high()),
        ([high(), high()], low()),
        ([high(), unknown()], unknown()),
        ([unknown(), low()], high()),
        ([unknown(), high()], unknown()),
        ([unknown(), unknown()], unknown()),
    ];

    for (input, output) in truth_table {
        check(&mut simulation, input, output);
    }
}

#[test]
fn test_xor2() {
    let mut simulation = Simulation::builder()
        .with_component("xor", xor2(["0", "1"], "z"))
        .with_io_many([input("0"), input("1"), output("z")])
        .build()
        .unwrap();

    let truth_table = [
        ([low(), low()], low()),
        ([low(), high()], high()),
        ([low(), unknown()], unknown()),
        ([high(), low()], high()),
        ([high(), high()], low()),
        ([high(), unknown()], unknown()),
        ([unknown(), low()], unknown()),
        ([unknown(), high()], unknown()),
        ([unknown(), unknown()], unknown()),
    ];

    for (input, output) in truth_table {
        check(&mut simulation, input, output);
    }
}

#[test]
fn test_half_adder() {
    let mut simulation = Simulation::<()>::builder()
        .with_component("half_adder", half_adder("0", "1", "sum", "carry"))
        .with_io_many([input("0"), input("1"), output("sum"), output("carry")])
        .build()
        .unwrap();

    let truth_table = [
        ([low(), low()], (low(), low())),
        ([low(), high()], (high(), low())),
        ([high(), low()], (high(), low())),
        ([high(), high()], (low(), high())),
    ];

    for (input, (sum, carry)) in truth_table {
        for (index, value) in input.into_iter().enumerate() {
            simulation.write_wire(&index.to_string(), value).unwrap();
        }
        simulation.settle();

        assert_eq!(simulation.read_wire("sum").unwrap(), sum);
        assert_eq!(simulation.read_wire("carry").unwrap(), carry);
    }
}

#[test]
fn test_full_adder() {
    let mut simulation = Simulation::<()>::builder()
        .with_component("full_adder", full_adder("0", "1", "2", "sum", "cout"))
        .with_io_many([
            input("0"),
            input("1"),
            input("2"),
            output("sum"),
            output("cout"),
        ])
        .build()
        .unwrap();

    let truth_table = [
        ([low(), low(), low()], (low(), low())),
        ([low(), low(), high()], (high(), low())),
        ([low(), high(), low()], (high(), low())),
        ([low(), high(), high()], (low(), high())),
        ([high(), low(), low()], (high(), low())),
        ([high(), low(), high()], (low(), high())),
        ([high(), high(), low()], (low(), high())),
        ([high(), high(), high()], (high(), high())),
    ];

    for (input, (sum, cout)) in truth_table {
        for (index, value) in input.into_iter().enumerate() {
            simulation.write_wire(&index.to_string(), value).unwrap();
        }
        simulation.settle();

        assert_eq!(simulation.read_wire("sum").unwrap(), sum);
        assert_eq!(simulation.read_wire("cout").unwrap(), cout);
    }
}

#[test]
fn test_xor4() {
    let mut simulation = Simulation::builder()
        .with_component("xor", xor4(["0", "1", "2", "3"], "z"))
        .with_io_many([input("0"), input("1"), input("2"), input("3"), output("z")])
        .build()
        .unwrap();

    let truth_table = [
        ([low(), low(), low(), low()], low()),
        ([high(), low(), low(), low()], high()),
        ([low(), high(), low(), low()], high()),
        ([low(), low(), high(), low()], high()),
        ([low(), low(), low(), high()], high()),
        ([high(), low(), high(), low()], low()),
        ([low(), high(), low(), high()], low()),
        ([high(), low(), low(), high()], low()),
        ([low(), high(), high(), low()], low()),
        ([high(), high(), high(), high()], low()),
        ([low(), high(), high(), high()], high()),
        ([high(), low(), high(), high()], high()),
        ([high(), high(), low(), high()], high()),
        ([high(), high(), high(), low()], high()),
    ];

    for (input, output) in truth_table {
        check(&mut simulation, input, output);
    }
}

#[test]
fn test_xnor2() {
    let mut simulation = Simulation::builder()
        .with_component("xor", xnor2(["0", "1"], "z"))
        .with_io_many([input("0"), input("1"), output("z")])
        .build()
        .unwrap();

    let truth_table = [
        ([low(), low()], high()),
        ([low(), high()], low()),
        ([low(), unknown()], unknown()),
        ([high(), low()], low()),
        ([high(), high()], high()),
        ([high(), unknown()], unknown()),
        ([unknown(), low()], unknown()),
        ([unknown(), high()], unknown()),
        ([unknown(), unknown()], unknown()),
    ];

    for (input, output) in truth_table {
        check(&mut simulation, input, output);
    }
}

fn assert_output(simulation: &Simulation, output_name: &str, expected: Signal) {
    assert_eq!(simulation.read_wire(output_name).unwrap(), expected);
}

fn latch_unknown_states(latch: &mut Simulation) {
    // 0 -> 1 with unknown enable

    // write the initial state
    latch.write_wire("data", low()).unwrap();
    latch.write_wire("enable", high()).unwrap();
    latch.settle();
    latch.write_wire("enable", low()).unwrap();
    latch.settle();

    assert_output(latch, "output", low());
    assert_output(latch, "output_inverse", high());

    // write the desired state
    latch.write_wire("data", high()).unwrap();
    latch.settle();

    // write unknown enable
    latch.write_wire("enable", unknown()).unwrap();
    latch.settle();

    // it should be unknown if it preserved the previous state or updated.
    assert_output(latch, "output", unknown());
    assert_output(latch, "output_inverse", unknown());

    // 1 -> 0 with unknown enable

    // write the initial state
    latch.write_wire("data", high()).unwrap();
    latch.write_wire("enable", high()).unwrap();
    latch.settle();
    latch.write_wire("enable", low()).unwrap();
    latch.settle();

    assert_output(latch, "output", high());
    assert_output(latch, "output_inverse", low());

    // write the desired state
    latch.write_wire("data", low()).unwrap();
    latch.settle();

    // write unknown enable
    latch.write_wire("enable", unknown()).unwrap();
    latch.settle();

    // it should be unknown if it preserved the previous state or updated.
    assert_output(latch, "output", unknown());
    assert_output(latch, "output_inverse", unknown());

    // 1 -> 1 with unknown enable

    // write the initial state
    latch.write_wire("data", high()).unwrap();
    latch.write_wire("enable", high()).unwrap();
    latch.settle();
    latch.write_wire("enable", low()).unwrap();
    latch.settle();

    assert_output(latch, "output", high());
    assert_output(latch, "output_inverse", low());

    // The data signal is still 1.

    // write unknown enable
    latch.write_wire("enable", unknown()).unwrap();
    latch.settle();

    // In a perfect world there is no reason for the output to change but for
    // the 4 NAND d latch implementation it would resolve to unknown.
    assert_output(latch, "output", unknown());
    assert_output(latch, "output_inverse", unknown());

    // 0 -> 0 with unknown enable

    // write the initial state
    latch.write_wire("data", low()).unwrap();
    latch.write_wire("enable", high()).unwrap();
    latch.settle();
    latch.write_wire("enable", low()).unwrap();
    latch.settle();

    assert_output(latch, "output", low());
    assert_output(latch, "output_inverse", high());

    // The data signal is still 0.

    // write unknown enable
    latch.write_wire("enable", unknown()).unwrap();
    latch.settle();

    assert_output(latch, "output", low());
    assert_output(latch, "output_inverse", high());
}

fn latch_regular_states(latch: &mut Simulation) {
    // write a 0 into the latch
    latch.write_wire("enable", high()).unwrap();
    latch.write_wire("data", low()).unwrap();
    latch.settle();

    assert_output(latch, "output", low());
    assert_output(latch, "output_inverse", high());

    // disable writing
    latch.write_wire("enable", low()).unwrap();
    latch.settle();

    assert_output(latch, "output", low());
    assert_output(latch, "output_inverse", high());

    // check that data is latched when enable is low
    latch.write_wire("data", high()).unwrap();
    latch.settle();

    assert_output(latch, "output", low());
    assert_output(latch, "output_inverse", high());

    // confirm that the data updates when enable is high
    latch.write_wire("enable", high()).unwrap();
    latch.settle();

    assert_output(latch, "output", high());
    assert_output(latch, "output_inverse", low());

    // confirm that the data is latched when enable is low
    latch.write_wire("enable", low()).unwrap();
    latch.settle();

    assert_output(latch, "output", high());
    assert_output(latch, "output_inverse", low());

    latch.write_wire("data", low()).unwrap();
    latch.settle();

    assert_output(latch, "output", high());
    assert_output(latch, "output_inverse", low());
}

/// Builds a gated D latch based on a NAND !SR latch.
/// https://en.wikipedia.org/wiki/Flip-flop_(electronics)#Gated_D_latch
fn gated_d_latch() -> Simulation {
    Simulation::builder()
        .with_io_many([
            input("data"),
            input("enable"),
            output("output"),
            output("output_inverse"),
        ])
        .with_component(
            "input_nand_1",
            nand2(["data", "enable"], "input_nand_1_out"),
        )
        .with_component(
            "input_nand_2",
            nand2(["input_nand_1_out", "enable"], "input_nand_2_out"),
        )
        .with_component(
            "sr_nand_1",
            nand2(["input_nand_1_out", "output_inverse"], "output"),
        )
        .with_component(
            "sr_nand_2",
            nand2(["output", "input_nand_2_out"], "output_inverse"),
        )
        .build()
        .unwrap()
}

#[test]
fn test_custom_d_latch() {
    let mut latch = gated_d_latch();
    latch.settle();

    latch_regular_states(&mut latch);
    latch_unknown_states(&mut latch);
}

#[test]
fn test_d_latch() {
    let mut latch = Simulation::builder()
        .with_io_many([
            input("data"),
            input("enable"),
            output("output"),
            output("output_inverse"),
        ])
        .with_component("dlatch", dlatch("data", "enable", "output"))
        .with_component("inverter", inverter("output", "output_inverse"))
        .build()
        .unwrap();

    latch.settle();

    latch_regular_states(&mut latch);
}

proptest! {
    #[test]
    fn test_aoi21(
        a: Signal,
        b1: Signal,
        b2: Signal,
    ) {
        let mut simulation = Simulation::<()>::builder()
            .with_io_many([
                input("a"),
                input("b1"),
                input("b2"),
                output("out"),
            ])
            .with_component("aoi", aoi21("a", "b1", "b2", "out"))
            .build()
            .unwrap();

        simulation.write_wire("a", a).unwrap();
        simulation.write_wire("b1", b1).unwrap();
        simulation.write_wire("b2", b2).unwrap();
        simulation.settle();

        assert_eq!(simulation.read_wire("out").unwrap(), a.or(b1.and(b2)).invert());
    }

    #[test]
    fn test_aoi22(
        a1: Signal,
        a2: Signal,
        b1: Signal,
        b2: Signal,
    ) {
        let mut simulation = Simulation::<()>::builder()
            .with_io_many([
                input("a1"),
                input("a2"),
                input("b1"),
                input("b2"),
                output("out"),
            ])
            .with_component("aoi", aoi22("a1", "a2", "b1", "b2", "out"))
            .build()
            .unwrap();

        simulation.write_wire("a1", a1).unwrap();
        simulation.write_wire("a2", a2).unwrap();
        simulation.write_wire("b1", b1).unwrap();
        simulation.write_wire("b2", b2).unwrap();
        simulation.settle();

        assert_eq!(simulation.read_wire("out").unwrap(), a1.and(a2).or(b1.and(b2)).invert());
    }

    #[test]
    fn test_aoi211(
        a: Signal,
        b: Signal,
        c1: Signal,
        c2: Signal,
    ) {
        let mut simulation = Simulation::<()>::builder()
            .with_io_many([
                input("a"),
                input("b"),
                input("c1"),
                input("c2"),
                output("out"),
            ])
            .with_component("aoi", aoi211("a", "b", "c1", "c2", "out"))
            .build()
            .unwrap();

        simulation.write_wire("a", a).unwrap();
        simulation.write_wire("b", b).unwrap();
        simulation.write_wire("c1", c1).unwrap();
        simulation.write_wire("c2", c2).unwrap();
        simulation.settle();

        assert_eq!(simulation.read_wire("out").unwrap(), a.or(b).or(c1.and(c2)).invert());
    }

    #[test]
    fn test_aoi221(
        a: Signal,
        b1: Signal,
        b2: Signal,
        c1: Signal,
        c2: Signal,
    ) {
        let mut simulation = Simulation::<()>::builder()
            .with_io_many([
                input("a"),
                input("b1"),
                input("b2"),
                input("c1"),
                input("c2"),
                output("out"),
            ])
            .with_component("aoi", aoi221("a", "b1", "b2", "c1", "c2", "out"))
            .build()
            .unwrap();

        simulation.write_wire("a", a).unwrap();
        simulation.write_wire("b1", b1).unwrap();
        simulation.write_wire("b2", b2).unwrap();
        simulation.write_wire("c1", c1).unwrap();
        simulation.write_wire("c2", c2).unwrap();
        simulation.settle();

        assert_eq!(simulation.read_wire("out").unwrap(), a.or(b1.and(b2)).or(c1.and(c2)).invert());
    }

    #[test]
    fn test_aoi222(
        a1: Signal,
        a2: Signal,
        b1: Signal,
        b2: Signal,
        c1: Signal,
        c2: Signal,
    ) {
        let mut simulation = Simulation::<()>::builder()
            .with_io_many([
                input("a1"),
                input("a2"),
                input("b1"),
                input("b2"),
                input("c1"),
                input("c2"),
                output("out"),
            ])
            .with_component("aoi", aoi222("a1", "a2", "b1", "b2", "c1", "c2", "out"))
            .build()
            .unwrap();

        simulation.write_wire("a1", a1).unwrap();
        simulation.write_wire("a2", a2).unwrap();
        simulation.write_wire("b1", b1).unwrap();
        simulation.write_wire("b2", b2).unwrap();
        simulation.write_wire("c1", c1).unwrap();
        simulation.write_wire("c2", c2).unwrap();
        simulation.settle();

        assert_eq!(simulation.read_wire("out").unwrap(), a1.and(a2).or(b1.and(b2)).or(c1.and(c2)).invert());
    }

    #[test]
    fn test_oai21(
        a: Signal,
        b1: Signal,
        b2: Signal,
    ) {
        let mut simulation = Simulation::<()>::builder()
            .with_io_many([
                input("a"),
                input("b1"),
                input("b2"),
                output("out"),
            ])
            .with_component("oai", oai21("a", "b1", "b2", "out"))
            .build()
            .unwrap();

        simulation.write_wire("a", a).unwrap();
        simulation.write_wire("b1", b1).unwrap();
        simulation.write_wire("b2", b2).unwrap();
        simulation.settle();

        assert_eq!(simulation.read_wire("out").unwrap(), a.and(b1.or(b2)).invert());
    }

    #[test]
    fn test_oai22(
        a1: Signal,
        a2: Signal,
        b1: Signal,
        b2: Signal,
    ) {
        let mut simulation = Simulation::<()>::builder()
            .with_io_many([
                input("a1"),
                input("a2"),
                input("b1"),
                input("b2"),
                output("out"),
            ])
            .with_component("oai", oai22("a1", "a2", "b1", "b2", "out"))
            .build()
            .unwrap();

        simulation.write_wire("a1", a1).unwrap();
        simulation.write_wire("a2", a2).unwrap();
        simulation.write_wire("b1", b1).unwrap();
        simulation.write_wire("b2", b2).unwrap();
        simulation.settle();

        assert_eq!(simulation.read_wire("out").unwrap(), a1.or(a2).and(b1.or(b2)).invert());
    }

    #[test]
    fn test_oai211(
        a: Signal,
        b: Signal,
        c1: Signal,
        c2: Signal,
    ) {
        let mut simulation = Simulation::<()>::builder()
            .with_io_many([
                input("a"),
                input("b"),
                input("c1"),
                input("c2"),
                output("out"),
            ])
            .with_component("oai", oai211("a", "b", "c1", "c2", "out"))
            .build()
            .unwrap();

        simulation.write_wire("a", a).unwrap();
        simulation.write_wire("b", b).unwrap();
        simulation.write_wire("c1", c1).unwrap();
        simulation.write_wire("c2", c2).unwrap();
        simulation.settle();

        assert_eq!(simulation.read_wire("out").unwrap(), a.and(b).and(c1.or(c2)).invert());
    }

    #[test]
    fn test_oai221(
        a: Signal,
        b1: Signal,
        b2: Signal,
        c1: Signal,
        c2: Signal,
    ) {
        let mut simulation = Simulation::<()>::builder()
            .with_io_many([
                input("a"),
                input("b1"),
                input("b2"),
                input("c1"),
                input("c2"),
                output("out"),
            ])
            .with_component("oai", oai221("a", "b1", "b2", "c1", "c2", "out"))
            .build()
            .unwrap();

        simulation.write_wire("a", a).unwrap();
        simulation.write_wire("b1", b1).unwrap();
        simulation.write_wire("b2", b2).unwrap();
        simulation.write_wire("c1", c1).unwrap();
        simulation.write_wire("c2", c2).unwrap();
        simulation.settle();

        assert_eq!(simulation.read_wire("out").unwrap(), a.and(b1.or(b2)).and(c1.or(c2)).invert());
    }

    #[test]
    fn test_oai222(
        a1: Signal,
        a2: Signal,
        b1: Signal,
        b2: Signal,
        c1: Signal,
        c2: Signal,
    ) {
        let mut simulation = Simulation::<()>::builder()
            .with_io_many([
                input("a1"),
                input("a2"),
                input("b1"),
                input("b2"),
                input("c1"),
                input("c2"),
                output("out"),
            ])
            .with_component("oai", oai222("a1", "a2", "b1", "b2", "c1", "c2", "out"))
            .build()
            .unwrap();

        simulation.write_wire("a1", a1).unwrap();
        simulation.write_wire("a2", a2).unwrap();
        simulation.write_wire("b1", b1).unwrap();
        simulation.write_wire("b2", b2).unwrap();
        simulation.write_wire("c1", c1).unwrap();
        simulation.write_wire("c2", c2).unwrap();
        simulation.settle();

        assert_eq!(simulation.read_wire("out").unwrap(), a1.or(a2).and(b1.or(b2)).and(c1.or(c2)).invert());
    }

    #[test]
    fn test_oai33(
        a1: Signal,
        a2: Signal,
        a3: Signal,
        b1: Signal,
        b2: Signal,
        b3: Signal,
    ) {
        let mut simulation = Simulation::<()>::builder()
            .with_io_many([
                input("a1"),
                input("a2"),
                input("a3"),
                input("b1"),
                input("b2"),
                input("b3"),
                output("out"),
            ])
            .with_component("oai", oai33("a1", "a2", "a3", "b1", "b2", "b3", "out"))
            .build()
            .unwrap();

        simulation.write_wire("a1", a1).unwrap();
        simulation.write_wire("a2", a2).unwrap();
        simulation.write_wire("a3", a3).unwrap();
        simulation.write_wire("b1", b1).unwrap();
        simulation.write_wire("b2", b2).unwrap();
        simulation.write_wire("b3", b3).unwrap();
        simulation.settle();

        assert_eq!(simulation.read_wire("out").unwrap(), a1.or(a2).or(a3).and(b1.or(b2).or(b3)).invert());
    }

    #[test]
    fn test_mux2(a: Signal, b: Signal, select: Signal) {
        let mut simulation = Simulation::<()>::builder()
            .with_io_many([
                input("a"),
                input("b"),
                input("select"),
                output("out"),
            ])
            .with_component("mux", mux2("a", "b", "select", "out"))
            .build()
            .unwrap();

        simulation.write_wire("a", a).unwrap();
        simulation.write_wire("b", b).unwrap();
        simulation.write_wire("select", select).unwrap();
        simulation.settle();

        match select {
            Signal::Low => assert_eq!(simulation.read_wire("out").unwrap(), a),
            Signal::High => assert_eq!(simulation.read_wire("out").unwrap(), b),
            Signal::Unknown => assert_eq!(simulation.read_wire("out").unwrap(), Signal::Unknown),
        }

    }

    #[test]
    fn test_full_adder_vs_manual(a: Signal, b: Signal, cin: Signal) {
        if a == Signal::Unknown || b == Signal::Unknown || cin == Signal::Unknown {
            return Ok(());
        }

        let mut primitive = Simulation::<()>::builder()
            .with_io_many([
                input("a"),
                input("b"),
                input("carry_in"),
                output("sum"),
                output("carry_out"),
            ])
            .with_component("full_adder", full_adder("a", "b", "carry_in", "sum", "carry_out"))
            .build()
            .unwrap();

        let mut compound = Simulation::builder()
            .with_io_many([
                input("a"),
                input("b"),
                input("carry_in"),
                output("sum"),
                output("carry_out"),
            ])
            .with_component("ha1", half_adder("a", "b", "ha1_sum", "ha1_carry"))
            .with_component("ha2", half_adder("carry_in", "ha1_sum", "sum", "ha2_carry"))
            .with_component("or", or2(["ha1_carry", "ha2_carry"], "carry_out"))
            .build()
            .unwrap();

        for simulation in [&mut primitive, &mut compound] {
            simulation.write_wire("a", a).unwrap();
            simulation.write_wire("b", b).unwrap();
            simulation.write_wire("carry_in", cin).unwrap();
            simulation.settle();
        }

        assert_eq!(primitive.read_wire("sum").unwrap(), compound.read_wire("sum").unwrap());
        assert_eq!(primitive.read_wire("carry_out").unwrap(), compound.read_wire("carry_out").unwrap());
    }
}
