use logic_simulation::{
    FaultOutOfBoundsError, Signal, Simulation,
    builder::{and2, full_adder, half_adder, input, inverter, output},
    fault::{FlipFault, StuckAtFault},
};
use proptest::prelude::*;

proptest! {
    #[test]
    fn faulty_and(a: Signal, b: Signal) {
        let mut sim = Simulation::builder()
            .with_io_many([input("a"), input("b"), output("z")])
            .with_component("and", and2(["a", "b"], "z"))
            .build().unwrap();

        sim.write_wire("a", a).unwrap();
        sim.write_wire("b", b).unwrap();

        assert_eq!(sim.make_faulty(1, StuckAtFault::High), Err(FaultOutOfBoundsError(1)));

        sim.make_faulty(0, StuckAtFault::Low).unwrap();
        sim.settle();
        assert_eq!(sim.read_wire("z").unwrap(), Signal::Low);

        sim.make_faulty(0, StuckAtFault::High).unwrap();
        sim.settle();
        assert_eq!(sim.read_wire("z").unwrap(), Signal::High);
    }


    #[test]
    fn faulty_and_flip(a: Signal, b: Signal, cin: Signal) {
        let mut sim = Simulation::builder()
            .with_io_many([input("a"), input("b"), output("z")])
            .with_component("and", and2(["a", "b"], "z"))
            .build().unwrap();

        sim.write_wire("a", a).unwrap();
        sim.write_wire("b", b).unwrap();

        sim.make_faulty(0, FlipFault).unwrap();
        sim.settle();
        assert_eq!(sim.read_wire("z").unwrap(), a.and(b).invert());

        sim.make_faulty(0, FlipFault).unwrap();
        sim.settle();
        assert_eq!(sim.read_wire("z").unwrap(), a.and(b).invert());
    }

    #[test]
    fn faulty_full_adder(a: Signal, b: Signal) {
        let mut sim = Simulation::builder()
            .with_io_many([input("a"), input("b"), input("cin"), output("sum"), output("cout")])
            .with_component("and", full_adder("a", "b", "cin", "sum", "cout"))
            .build().unwrap();

        sim.write_wire("a", a).unwrap();
        sim.write_wire("b", b).unwrap();

        sim.settle();
        let expected_sum = sim.read_wire("sum").unwrap();
        let expected_cout = sim.read_wire("cout").unwrap();

        sim.make_faulty(0, FlipFault).unwrap();
        sim.settle();

        assert_eq!(sim.read_wire("sum").unwrap(), expected_sum.invert());
        assert_eq!(sim.read_wire("cout").unwrap(), expected_cout);

        sim.make_faulty(1, FlipFault).unwrap();
        sim.settle();

        assert_eq!(sim.read_wire("sum").unwrap(), expected_sum);
        assert_eq!(sim.read_wire("cout").unwrap(), expected_cout.invert());

        assert_eq!(sim.make_faulty(2, FlipFault), Err(FaultOutOfBoundsError(2)));

    }

    #[test]
    fn faulty_isolated_components(
        a in prop_oneof![Just(Signal::Low), Just(Signal::High)],
        b in prop_oneof![Just(Signal::Low), Just(Signal::High)],
        c in prop_oneof![Just(Signal::Low), Just(Signal::High)],
        d in prop_oneof![Just(Signal::Low), Just(Signal::High)],
    ) {
        // HalfAdder (2 outputs) and AND2 (1 output) drive three independent
        // outputs. Each fault target must affect exactly one of them, and the
        // three fault indices together must cover all three outputs.
        let mut sim = Simulation::builder()
            .with_io_many([
                input("a"), input("b"), input("c"), input("d"),
                output("sum"), output("carry"), output("z"),
            ])
            .with_component("ha", half_adder("a", "b", "sum", "carry"))
            .with_component("and", and2(["c", "d"], "z"))
            .build().unwrap();

        sim.write_wire("a", a).unwrap();
        sim.write_wire("b", b).unwrap();
        sim.write_wire("c", c).unwrap();
        sim.write_wire("d", d).unwrap();
        sim.settle();

        let expected_sum = sim.read_wire("sum").unwrap();
        let expected_carry = sim.read_wire("carry").unwrap();
        let expected_z = sim.read_wire("z").unwrap();

        let mut affected = Vec::new();
        for idx in 0..3 {
            sim.make_faulty(idx, FlipFault).unwrap();
            sim.settle();

            let sum = sim.read_wire("sum").unwrap();
            let carry = sim.read_wire("carry").unwrap();
            let z = sim.read_wire("z").unwrap();

            let changed: Vec<&str> = [
                ("sum", sum != expected_sum),
                ("carry", carry != expected_carry),
                ("z", z != expected_z),
            ]
            .into_iter()
            .filter_map(|(name, c)| c.then_some(name))
            .collect();

            assert_eq!(changed.len(), 1, "fault {idx} affected {changed:?}, expected exactly one output");
            affected.push(changed[0]);
        }

        affected.sort();
        assert_eq!(affected, vec!["carry", "sum", "z"]);

        assert_eq!(sim.make_faulty(3, FlipFault), Err(FaultOutOfBoundsError(3)));
    }

    #[test]
    fn faulty_inverter_chain(a in prop_oneof![Just(Signal::Low), Just(Signal::High)]) {
        // Three chained inverters: `out = NOT(NOT(NOT a)) = NOT a`. Flipping
        // any one inverter's output toggles the parity of the chain, so the
        // final output must equal `a`. Two of the three fault targets sit on
        // internal wires, so the effect can only reach `out` by propagating
        // through the remaining downstream inverters.
        let mut sim = Simulation::builder()
            .with_io_many([input("a"), output("out")])
            .with_component("inv1", inverter("a", "m1"))
            .with_component("inv2", inverter("m1", "m2"))
            .with_component("inv3", inverter("m2", "out"))
            .build().unwrap();

        sim.write_wire("a", a).unwrap();
        sim.settle();
        let expected = sim.read_wire("out").unwrap();
        assert_eq!(expected, a.invert());

        for idx in 0..3 {
            sim.make_faulty(idx, FlipFault).unwrap();
            sim.settle();
            assert_eq!(sim.read_wire("out").unwrap(), a,
                "fault index {idx} did not propagate to the output");
        }

        assert_eq!(sim.make_faulty(3, FlipFault), Err(FaultOutOfBoundsError(3)));
    }
}

#[test]
fn fault_radix_mixed_components() {
    let mut sim = Simulation::builder()
        .with_io_many([
            input("a"),
            input("b"),
            input("c"),
            output("sum"),
            output("carry"),
            output("not_c"),
        ])
        .with_component("ha", half_adder("a", "b", "sum", "carry"))
        .with_component("inv", inverter("c", "not_c"))
        .build()
        .unwrap();

    assert_eq!(sim.fault_radix(), 3);

    for idx in 0..3 {
        sim.make_faulty(idx, FlipFault).unwrap();
    }
    assert_eq!(sim.make_faulty(3, FlipFault), Err(FaultOutOfBoundsError(3)));
}
