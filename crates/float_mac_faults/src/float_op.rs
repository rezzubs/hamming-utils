use ariadne::sources;
use eyre::{Result, WrapErr, bail, eyre};
use rand::random;
use std::{
    ops::{Deref, DerefMut},
    path::{Path, PathBuf},
    time::Instant,
};

use logic_simulation::{
    Signal, Simulation, bus::BusVectorMismatchError, fault::Fault, netlist::parse_simulation,
};

use crate::TestArgs;

const F32_BITS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BitResult {
    /// The result matches the reference.
    Match,
    /// The result flipped
    Flip,
    /// The result used to be known but is now unknown.
    Unknown,
}

impl std::fmt::Display for BitResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BitResult::Match => write!(f, "0"),
            BitResult::Flip => write!(f, "1"),
            BitResult::Unknown => write!(f, "X"),
        }
    }
}

/// A bitwise comparison of float computation output signals.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BitResults(pub [BitResult; F32_BITS]);

impl BitResults {
    /// Check if all bits match.
    pub fn all_match(&self) -> bool {
        self.iter().all(|r| r == &BitResult::Match)
    }

    /// Check if any bits are unknown.
    pub fn contains_unknown(&self) -> bool {
        self.iter().any(|r| r == &BitResult::Unknown)
    }
}

impl Deref for BitResults {
    type Target = [BitResult];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for BitResults {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl std::fmt::Display for BitResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for bit in self.0.iter().rev() {
            write!(f, "{}", bit)?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Bits(pub [Signal; F32_BITS]);

impl Bits {
    /// See how the value of `other` compares to `self`.
    ///
    /// # Panics
    ///
    /// Panics if `self` contains any `Unknown` signals.
    pub fn compare(&self, other: &Self) -> BitResults {
        let mut results = [BitResult::Match; F32_BITS];
        for ((result, a), b) in results.iter_mut().zip(self.0).zip(other.0) {
            match (a, b) {
                (Signal::Unknown, _) => {
                    panic!("Unknown signal in reference implementation: {}", self)
                }
                (Signal::High, Signal::Low) | (Signal::Low, Signal::High) => {
                    *result = BitResult::Flip
                }
                (_, Signal::Unknown) => *result = BitResult::Unknown,
                _ => {}
            }
        }
        BitResults(results)
    }
}

impl std::fmt::Display for Bits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for bit in self.0.iter().rev() {
            write!(f, "{}", bit)?;
        }
        Ok(())
    }
}

impl Deref for Bits {
    type Target = [Signal];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Bits {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<f32> for Bits {
    fn from(value: f32) -> Self {
        let mut bits = [Signal::Unknown; F32_BITS];
        let mut value_bits = value.to_bits();
        for bit in bits.iter_mut() {
            *bit = if value_bits & 1 == 1 {
                Signal::High
            } else {
                Signal::Low
            };
            value_bits >>= 1;
        }

        Self(bits)
    }
}

/// A reference implementation to compare a [`Logic`] instance against.
pub trait Reference {
    fn compute(&self, input1: f32, input2: f32) -> Bits;
}

pub struct Multiply;

impl Reference for Multiply {
    fn compute(&self, input1: f32, input2: f32) -> Bits {
        Bits::from(input1 * input2)
    }
}

pub struct Add;

impl Reference for Add {
    fn compute(&self, input1: f32, input2: f32) -> Bits {
        Bits::from(input1 + input2)
    }
}

#[derive(Debug)]
pub struct Logic<F> {
    pub kind: String,
    pub simulation: Simulation<F>,
    pub input1: String,
    pub input2: String,
    pub output: String,
    pub source: PathBuf,
}

impl<F> Logic<F>
where
    F: Fault,
{
    pub fn load(
        path: PathBuf,
        input1: String,
        input2: String,
        output: String,
        kind: String,
    ) -> Result<Self> {
        let src = std::fs::read_to_string(&path)
            .wrap_err(format!("couldn't read {} to a string", path.display()))?;

        let span_path = path.to_string_lossy().to_string();
        let simulation = match parse_simulation(&src, span_path.clone()) {
            Ok(simulation) => simulation,
            Err(reports) => {
                for report in reports {
                    report
                        .eprint(sources([(span_path.clone(), &src)]))
                        .wrap_err("failed to print report")?;
                }

                bail!("failed to load {} from {}", kind, path.display())
            }
        };

        Ok(Self {
            simulation,
            input1,
            input2,
            output,
            source: path,
            kind,
        })
    }

    pub fn compute<T>(&mut self, input1: T, input2: T) -> Result<Bits>
    where
        T: Into<Bits>,
    {
        let mut input1_bus = self.simulation.input_bus(&self.input1).wrap_err(format!(
            "{} is not a valid input value for the {} netlist at {}",
            self.input1,
            self.kind,
            self.source.display(),
        ))?;

        let range = input1_bus.range();
        if range.len() != F32_BITS {
            bail!(
                "input1 {} doesn't have a 32 bit range in the {} netlist at {}",
                self.input1,
                self.kind,
                self.source.display()
            );
        };

        input1_bus
            .write_vector(range, &input1.into().0)
            .unwrap_or_else(|err| match err {
                BusVectorMismatchError::RangeMismatch { .. }
                | BusVectorMismatchError::InvalidRange { .. } => {
                    unreachable!("the range is known to be correct and f32 fits in 32 bits")
                }
            });

        let mut input2_bus = self.simulation.input_bus(&self.input2).wrap_err(format!(
            "{} is not a valid input value for the {} netlist at {}",
            self.input2,
            self.kind,
            self.source.display(),
        ))?;

        let range = input2_bus.range();
        if range.len() != F32_BITS {
            bail!(
                "input2 {} doesn't have a 32 bit range in the {} netlist at {}",
                self.input2,
                self.kind,
                self.source.display()
            );
        };

        input2_bus
            .write_vector(range, &input2.into().0)
            .unwrap_or_else(|err| match err {
                BusVectorMismatchError::RangeMismatch { .. }
                | BusVectorMismatchError::InvalidRange { .. } => {
                    unreachable!("the range is known to be correct and f32 fits in 32 bits")
                }
            });

        self.simulation.settle();

        let output_bus = self.simulation.output_bus(&self.output).wrap_err(format!(
            "{} is not a valid output value for the {} netlist at {}",
            self.output,
            self.kind,
            self.source.display(),
        ))?;

        let range = output_bus.range();

        if range.len() != F32_BITS {
            bail!(
                "output {} doesn't have a 32 bit range in the {} netlist at {}",
                self.output,
                self.kind,
                self.source.display()
            );
        }

        let raw_output = output_bus
            .read_vector(range)
            .unwrap_or_else(|err| match err {
                BusVectorMismatchError::RangeMismatch { .. }
                | BusVectorMismatchError::InvalidRange { .. } => {
                    unreachable!("the range is known to be correct and f32 fits in 32 bits")
                }
            });
        assert_eq!(raw_output.len(), F32_BITS);

        let mut output = [Signal::Unknown; F32_BITS];
        for (i, (raw, out)) in raw_output.into_iter().zip(output.iter_mut()).enumerate() {
            *out = raw.ok_or(eyre!("{}[{}] didn't produce a signal", self.output, i))?;
        }

        Ok(Bits(output))
    }

    /// Test a computation against a reference implementation.
    pub fn test(
        &mut self,
        input1: f32,
        input2: f32,
        reference: &impl Reference,
    ) -> Result<BitResults> {
        let expected = reference.compute(input1, input2);
        let actual = self.compute(input1, input2)?;

        Ok(actual.compare(&expected))
    }
}

pub fn test_logic(
    args: TestArgs,
    kind: &str,
    reference: &impl Reference,
    setup: impl FnOnce(&mut Simulation) -> Result<()>,
) -> eyre::Result<()> {
    let mut logic = Logic::<()>::load(
        args.source,
        args.input1,
        args.input2,
        args.output,
        kind.to_owned(),
    )?;

    setup(&mut logic.simulation).wrap_err(format!(
        "failed to set up {} at {}",
        kind,
        logic.source.display(),
    ))?;

    let mut correct = 0;
    let mut incorrect = 0;
    let mut checkpoint = Instant::now();
    let timeout = std::time::Duration::from_secs_f64(args.print_frequency);

    loop {
        if checkpoint.elapsed() > timeout {
            let ratio = correct as f64 / (correct + incorrect) as f64;
            println!(
                "{} correct, {} incorrect, {:00.2}%",
                correct,
                incorrect,
                ratio * 100.
            );
            checkpoint = Instant::now();
        }

        let input1 = f32::from_bits(random());
        let input2 = f32::from_bits(random());

        let result = logic.test(input1, input2, reference)?;

        if result.all_match() {
            correct += 1;
        } else {
            incorrect += 1;
        }
    }
}

pub fn configure_multiplier_rounding<F>(
    round_name: String,
    source: &Path,
    sim: &mut Simulation<F>,
) -> Result<()> {
    let kind = "multiplier";
    let mut round_bus = sim.input_bus(&round_name).wrap_err(format!(
        "{} is not a valid input value for the {} netlist at {}",
        round_name,
        kind,
        source.display(),
    ))?;

    let range = round_bus.range();
    if range.len() != 3 {
        bail!("expected {} to have a 3 bit range", round_name);
    }

    round_bus
        .write_vector(range, &[Signal::Low, Signal::Low, Signal::Low])
        .unwrap_or_else(|err| match err {
            BusVectorMismatchError::RangeMismatch { .. }
            | BusVectorMismatchError::InvalidRange { .. } => {
                unreachable!("the range is known to be correct")
            }
        });

    Ok(())
}
