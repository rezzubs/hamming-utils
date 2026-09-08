mod app;
mod float_op;

use crate::{
    app::App,
    float_op::{Add, Multiply},
};
use clap::{Args, Parser, Subcommand};
use eyre::Result;
use std::{path::PathBuf, time::Duration};

fn main() -> Result<()> {
    color_eyre::install().expect("failed to install color_eyre");

    let cli = Cli::parse();

    match cli.command {
        Command::TestAdd(args) => {
            float_op::test_logic(args, "adder", &Add, |_| Ok(()))?;
        }
        Command::TestMul {
            common: args,
            round,
        } => {
            let kind = "multiplier";
            float_op::test_logic(args.clone(), kind, &Multiply, |sim| {
                float_op::configure_multiplier_rounding(round, &args.source, sim)
            })?;
        }
        Command::Faults { args } => {
            let native_options = eframe::NativeOptions::default();
            eframe::run_native(
                "Float MAC fault simulation",
                native_options,
                Box::new(|cc| {
                    Ok(Box::new(App::new(
                        cc,
                        args,
                        Duration::from_secs_f64(1. / 60.),
                    )?))
                }),
            )?;
        }
    }

    Ok(())
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Parse a verilog netlist containing an adder module and test it against a cpu.
    TestAdd(TestArgs),
    /// Parse a verilog netlist containing a multiplier module and test it against a cpu.
    TestMul {
        #[command(flatten)]
        common: TestArgs,
        #[arg(long)]
        round: String,
    },
    Faults {
        #[command(flatten)]
        args: AppArgs,
    },
}

#[derive(Clone, Args)]
struct AppArgs {
    #[arg(long)]
    mul_source: PathBuf,
    #[arg(long)]
    mul_input1: String,
    #[arg(long)]
    mul_input2: String,
    #[arg(long)]
    mul_round: String,
    #[arg(long)]
    mul_output: String,
    #[arg(long)]
    add_source: PathBuf,
    #[arg(long)]
    add_input1: String,
    #[arg(long)]
    add_input2: String,
    #[arg(long)]
    add_output: String,
}

#[derive(Clone, Args)]
struct TestArgs {
    /// The path to the netlist.
    source: PathBuf,
    /// The name of the first input port.
    #[arg(long)]
    input1: String,
    /// The name of the second input port.
    #[arg(long)]
    input2: String,
    /// The name of the output port.
    #[arg(long)]
    output: String,
    /// The frequency in seconds at which to print results.
    #[arg(long, default_value_t = 2.0)]
    print_frequency: f64,
}
