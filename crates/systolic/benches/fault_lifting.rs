use ndarray::Array2;
use systolic::{
    Index2,
    array::SystolicArray,
    fault::{
        PeFaultRegister, PeRegisterFault, RegisterFault, RegisterHook, StuckAt, TargetedFault,
    },
};

/// Combined array-size and batch-size configuration used to parametrize each
/// benchmark group. Both `literal` and `lifted` variants receive the same
/// configs so their results are directly comparable.
#[derive(Clone, Copy)]
struct Config {
    array_size: usize,
    batch_size: usize,
}

impl std::fmt::Display for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "arr{}x{}_batch{}", self.array_size, self.array_size, self.batch_size)
    }
}

/// Single-pass scenarios: the weight matrix exactly fills the array, so the
/// mapping has one pass. Covers small (8x8) and large (64x64) arrays.
fn single_pass_configs() -> Vec<Config> {
    [8usize, 64]
        .iter()
        .flat_map(|&array_size| {
            [1usize, 64, 256].iter().map(move |&batch_size| Config { array_size, batch_size })
        })
        .collect()
}

/// Multi-pass configs: 8x8 array with a 64x64 weight matrix, producing 64
/// passes. Literal must simulate each pass through the array separately;
/// lifted collapses all passes into one dense matmul plus a small fix-up.
fn multi_pass_configs() -> Vec<Config> {
    [1usize, 64, 256]
        .iter()
        .map(|&batch_size| Config { array_size: 8, batch_size })
        .collect()
}

fn targeted_fault(
    nrows: usize,
    ncols: usize,
    register: PeFaultRegister,
) -> TargetedFault<PeRegisterFault> {
    TargetedFault {
        target: Index2 { x: (ncols / 2) as u16, y: (nrows / 2) as u16 },
        fault: PeRegisterFault {
            register,
            fault: RegisterFault { stuck_at: StuckAt::One, bit_index: 0 },
        },
    }
}

mod weight_fault {
    use super::*;

    #[divan::bench(args = single_pass_configs())]
    fn literal(bencher: divan::Bencher, config: Config) {
        let Config { array_size, batch_size } = config;
        let array = SystolicArray::<f32>::new(array_size, array_size).unwrap();
        let weights = Array2::<f32>::ones((array_size, array_size));
        let activations = Array2::<f32>::ones((array_size, batch_size));
        let mapping = array.auto_mapping_for(&weights);
        let hook = RegisterHook::from_fault(targeted_fault(array_size, array_size, PeFaultRegister::Weight));

        bencher
            .with_inputs(|| array.clone().with_hook(hook.clone()))
            .bench_local_refs(|faulty| {
                divan::black_box(faulty.matmul(&mapping, &weights, &activations))
            });
    }

    #[divan::bench(args = single_pass_configs())]
    fn lifted(bencher: divan::Bencher, config: Config) {
        let Config { array_size, batch_size } = config;
        let array = SystolicArray::<f32>::new(array_size, array_size).unwrap();
        let weights = Array2::<f32>::ones((array_size, array_size));
        let activations = Array2::<f32>::ones((array_size, batch_size));
        let mapping = array.auto_mapping_for(&weights);
        let lifted = mapping.lift_register_fault(&targeted_fault(array_size, array_size, PeFaultRegister::Weight));

        bencher
            .with_inputs(|| (weights.clone(), activations.clone()))
            .bench_values(|(weights, activations)| {
                divan::black_box(lifted.matmul(weights, activations))
            });
    }
}

mod activation_fault {
    use super::*;

    #[divan::bench(args = single_pass_configs())]
    fn literal(bencher: divan::Bencher, config: Config) {
        let Config { array_size, batch_size } = config;
        let array = SystolicArray::<f32>::new(array_size, array_size).unwrap();
        let weights = Array2::<f32>::ones((array_size, array_size));
        let activations = Array2::<f32>::ones((array_size, batch_size));
        let mapping = array.auto_mapping_for(&weights);
        let hook = RegisterHook::from_fault(targeted_fault(array_size, array_size, PeFaultRegister::Activation));

        bencher
            .with_inputs(|| array.clone().with_hook(hook.clone()))
            .bench_local_refs(|faulty| {
                divan::black_box(faulty.matmul(&mapping, &weights, &activations))
            });
    }

    #[divan::bench(args = single_pass_configs())]
    fn lifted(bencher: divan::Bencher, config: Config) {
        let Config { array_size, batch_size } = config;
        let array = SystolicArray::<f32>::new(array_size, array_size).unwrap();
        let weights = Array2::<f32>::ones((array_size, array_size));
        let activations = Array2::<f32>::ones((array_size, batch_size));
        let mapping = array.auto_mapping_for(&weights);
        let lifted = mapping.lift_register_fault(&targeted_fault(array_size, array_size, PeFaultRegister::Activation));

        bencher
            .with_inputs(|| (weights.clone(), activations.clone()))
            .bench_values(|(weights, activations)| {
                divan::black_box(lifted.matmul(weights, activations))
            });
    }
}

mod accumulator_fault {
    use super::*;

    #[divan::bench(args = single_pass_configs())]
    fn literal(bencher: divan::Bencher, config: Config) {
        let Config { array_size, batch_size } = config;
        let array = SystolicArray::<f32>::new(array_size, array_size).unwrap();
        let weights = Array2::<f32>::ones((array_size, array_size));
        let activations = Array2::<f32>::ones((array_size, batch_size));
        let mapping = array.auto_mapping_for(&weights);
        let hook = RegisterHook::from_fault(targeted_fault(array_size, array_size, PeFaultRegister::Accumulator));

        bencher
            .with_inputs(|| array.clone().with_hook(hook.clone()))
            .bench_local_refs(|faulty| {
                divan::black_box(faulty.matmul(&mapping, &weights, &activations))
            });
    }

    #[divan::bench(args = single_pass_configs())]
    fn lifted(bencher: divan::Bencher, config: Config) {
        let Config { array_size, batch_size } = config;
        let array = SystolicArray::<f32>::new(array_size, array_size).unwrap();
        let weights = Array2::<f32>::ones((array_size, array_size));
        let activations = Array2::<f32>::ones((array_size, batch_size));
        let mapping = array.auto_mapping_for(&weights);
        let lifted = mapping.lift_register_fault(&targeted_fault(array_size, array_size, PeFaultRegister::Accumulator));

        bencher
            .with_inputs(|| (weights.clone(), activations.clone()))
            .bench_values(|(weights, activations)| {
                divan::black_box(lifted.matmul(weights, activations))
            });
    }
}

/// Multi-pass scenario: 8x8 array, 64x64 weight matrix, 64 passes.
///
/// This is where the gap between literal and lifted is most dramatic. Literal
/// must simulate all 64 passes through the array; lifted still does one dense
/// 64x64 matmul plus a small accumulator fix-up regardless of pass count.
mod multi_pass {
    use super::*;

    const ARRAY_SIZE: usize = 8;
    const WEIGHT_SIZE: usize = 64;

    #[divan::bench(args = multi_pass_configs())]
    fn literal(bencher: divan::Bencher, config: Config) {
        let Config { array_size: _, batch_size } = config;
        let array = SystolicArray::<f32>::new(ARRAY_SIZE, ARRAY_SIZE).unwrap();
        let weights = Array2::<f32>::ones((WEIGHT_SIZE, WEIGHT_SIZE));
        let activations = Array2::<f32>::ones((WEIGHT_SIZE, batch_size));
        let mapping = array.auto_mapping_for(&weights);
        let hook = RegisterHook::from_fault(targeted_fault(ARRAY_SIZE, ARRAY_SIZE, PeFaultRegister::Accumulator));

        bencher
            .with_inputs(|| array.clone().with_hook(hook.clone()))
            .bench_local_refs(|faulty| {
                divan::black_box(faulty.matmul(&mapping, &weights, &activations))
            });
    }

    #[divan::bench(args = multi_pass_configs())]
    fn lifted(bencher: divan::Bencher, config: Config) {
        let Config { array_size: _, batch_size } = config;
        let array = SystolicArray::<f32>::new(ARRAY_SIZE, ARRAY_SIZE).unwrap();
        let weights = Array2::<f32>::ones((WEIGHT_SIZE, WEIGHT_SIZE));
        let activations = Array2::<f32>::ones((WEIGHT_SIZE, batch_size));
        let mapping = array.auto_mapping_for(&weights);
        let lifted = mapping.lift_register_fault(&targeted_fault(ARRAY_SIZE, ARRAY_SIZE, PeFaultRegister::Accumulator));

        bencher
            .with_inputs(|| (weights.clone(), activations.clone()))
            .bench_values(|(weights, activations)| {
                divan::black_box(lifted.matmul(weights, activations))
            });
    }
}

fn main() {
    divan::main();
}
