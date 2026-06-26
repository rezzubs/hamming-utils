use std::{
    collections::HashMap,
    ops::Add,
    time::{Duration, Instant},
};

use egui_plot::{Bar, BarChart};
use eyre::Result;
use logic_simulation::fault::StuckAtFault;
use rand::{random, random_range};

use crate::{
    AppArgs,
    float_op::{self, BitResult, BitResults, Bits, configure_multiplier_rounding},
};

type Logic = float_op::Logic<StuckAtFault>;

/// One for stuck-at-0, one for stuck-at-1.
const FAULT_STATES: usize = 2;

#[derive(Debug)]
struct MultiplyAdd {
    multiplier: Logic,
    adder: Logic,
}

impl MultiplyAdd {
    fn compute<T>(&mut self, multiply1: T, multiply2: T, add: T) -> Result<Bits>
    where
        T: Into<Bits>,
    {
        let multiply_result = self.multiplier.compute(multiply1, multiply2)?;
        let add_result = self.adder.compute(multiply_result, add.into())?;

        Ok(add_result)
    }

    /// Return the maximum number of different possible faults.
    fn fault_radix(&mut self) -> usize {
        FAULT_STATES * (self.mul_radix() + self.add_radix())
    }

    fn mul_radix(&mut self) -> usize {
        self.multiplier.simulation.fault_radix()
    }

    fn add_radix(&mut self) -> usize {
        self.adder.simulation.fault_radix()
    }

    fn make_faulty(&mut self, target: usize) {
        if target >= self.fault_radix() {
            panic!("invalid fault target: {} >= {}", target, self.fault_radix())
        }

        self.remove_fault();

        let faults_per_state = self.mul_radix() + self.add_radix();
        let fault_kind = target / faults_per_state;
        debug_assert!(fault_kind < FAULT_STATES);
        let remainder = target % faults_per_state;

        let (multiplier_fault, adder_fault) = if remainder >= self.add_radix() {
            let fault = remainder - self.add_radix();
            debug_assert!(fault < self.mul_radix());
            (Some(fault), None)
        } else {
            (None, Some(remainder))
        };

        let fault_kind = match fault_kind {
            0 => StuckAtFault::Low,
            1 => StuckAtFault::High,
            _ => unreachable!("we checked the range at the start"),
        };

        // these branches are mutually exclusive
        if let Some(multiplier_fault) = multiplier_fault {
            self.multiplier
                .simulation
                .make_faulty(multiplier_fault, fault_kind)
                .expect("we checked the range at the start");
        }
        if let Some(adder_fault) = adder_fault {
            self.adder
                .simulation
                .make_faulty(adder_fault, fault_kind)
                .expect("we checked the range at the start");
        }
    }

    fn remove_fault(&mut self) {
        self.multiplier.simulation.remove_fault();
        self.adder.simulation.remove_fault();
    }
}

#[derive(Debug, PartialEq, Eq)]
enum RunState {
    Running,
    Paused,
}

#[derive(Debug, PartialEq, Eq)]
enum ViewMode {
    Patterns,
    Bits,
}

fn add_array<const N: usize, T>(a: [T; N], b: [T; N]) -> [T; N]
where
    T: Add<T, Output = T> + Default + Copy,
{
    let mut result = [T::default(); N];
    for i in 0..N {
        result[i] = a[i] + b[i];
    }
    result
}

#[derive(Debug)]
pub struct App {
    logic: MultiplyAdd,
    results: HashMap<BitResults, u32>,

    step_requested: bool,
    target_frame_time: Duration,
    previous_frame_render_time: Duration,
    runs: u32,

    run_state: RunState,
    view_mode: ViewMode,

    bars_to_skip: usize,
    bar_limit: usize,
}

impl App {
    pub fn new(
        _cc: &eframe::CreationContext<'_>,
        args: AppArgs,
        target_frame_time: Duration,
    ) -> Result<Self> {
        let adder = Logic::load(
            args.add_source.clone(),
            args.add_input1,
            args.add_input2,
            args.add_output,
            "adder".to_owned(),
        )?;
        let mut multiplier = Logic::load(
            args.mul_source,
            args.mul_input1,
            args.mul_input2,
            args.mul_output,
            "multiplier".to_owned(),
        )?;
        configure_multiplier_rounding(
            args.mul_round,
            &multiplier.source,
            &mut multiplier.simulation,
        )?;

        let logic = MultiplyAdd { multiplier, adder };

        Ok(Self {
            logic,
            results: HashMap::new(),
            step_requested: false,
            run_state: RunState::Paused,
            target_frame_time,
            previous_frame_render_time: Duration::ZERO,
            runs: 0,
            bars_to_skip: 0,
            bar_limit: 1,
            view_mode: ViewMode::Patterns,
        })
    }

    fn step(&mut self) {
        self.runs += 1;

        let update_limit = self.bar_limit == self.results.len();

        let a = random::<f32>();
        let b = random::<f32>();
        let c = random::<f32>();
        let fault = random_range(0..self.logic.fault_radix());

        self.logic.remove_fault();
        let expected = self
            .logic
            .compute(a, b, c)
            .expect("logic has been checked and should not fail after");
        self.logic.make_faulty(fault);
        let faulty = self
            .logic
            .compute(a, b, c)
            .expect("logic has been checked and should not fail after");

        let result = expected.compare(&faulty);

        self.results
            .entry(result)
            .and_modify(|occurences| *occurences += 1)
            .or_insert(1);

        if update_limit {
            self.bar_limit = self.results.len();
        }
    }

    fn draw_pattern_bars(&mut self, plot_ui: &mut egui_plot::PlotUi<'_>) {
        let mut results = self
            .results
            .iter()
            .map(|(a, b)| (a.clone(), *b))
            .collect::<Vec<_>>();
        results.sort_by_key(|(_, occurances)| std::cmp::Reverse(*occurances));
        let total = results
            .iter()
            .map(|(_, occurances)| *occurances as f64)
            .sum::<f64>();

        let bars = results
            .into_iter()
            .skip(self.bars_to_skip)
            .take(self.bar_limit)
            .enumerate()
            .map(|(index, (result, occurences))| {
                let mut bar =
                    Bar::new(index as f64, occurences as f64 / total).name(result.to_string());

                if result.contains_unknown() {
                    bar = bar.fill(egui::Color32::GREEN);
                }

                bar
            })
            .collect();
        let bar_chart = BarChart::new("results", bars).width(1.);
        plot_ui.bar_chart(bar_chart);
    }

    fn view_selector(&mut self, ui: &mut egui::Ui) {
        egui::ComboBox::from_label("view")
            .selected_text(format!("{:?}", self.view_mode))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.view_mode, ViewMode::Patterns, "Patterns");
                ui.selectable_value(&mut self.view_mode, ViewMode::Bits, "Bits");
            });
    }

    fn draw_header(&mut self, ui: &mut egui::Ui) -> egui::InnerResponse<()> {
        ui.horizontal(|ui| {
            self.view_selector(ui);
            match self.run_state {
                RunState::Running => {
                    if ui.button("pause").clicked() {
                        self.run_state = RunState::Paused;
                    };
                }
                RunState::Paused => {
                    if ui.button("run").clicked() {
                        self.run_state = RunState::Running;
                    };
                }
            }
            self.step_requested = ui.button("step").clicked();
            if self.view_mode == ViewMode::Patterns {
                ui.add(
                    egui::Slider::new(&mut self.bars_to_skip, 0..=self.results.len() - 1)
                        .text("Skip")
                        .logarithmic(true),
                );
                ui.add(
                    egui::Slider::new(&mut self.bar_limit, 1..=self.results.len())
                        .text("Limit")
                        .logarithmic(true),
                );
                ui.label(format!("Runs: {}", self.runs));
            }
        })
    }

    fn draw_bits(&mut self, plot_ui: &mut egui_plot::PlotUi<'_>) {
        let bit_counts = self.results.iter().fold([0; 32], |acc, (pattern, count)| {
            let mut pattern_bit_counts = [0; 32];
            for (i, bit) in pattern_bit_counts.iter_mut().enumerate() {
                if pattern.0[i] != BitResult::Match {
                    *bit = *count;
                }
            }

            add_array(acc, pattern_bit_counts)
        });

        let total = bit_counts.iter().sum::<u32>();

        let bars = bit_counts
            .into_iter()
            .enumerate()
            .map(|(i, count)| {
                Bar::new(i as f64, count as f64 / total as f64)
                    .name(format!("Bit {} - {} times", i, count))
            })
            .collect();

        plot_ui.bar_chart(BarChart::new("Bit Distribution", bars).width(1.));
    }
}

impl eframe::App for App {
    fn logic(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        _ = frame;
        let start = Instant::now();

        if self.step_requested {
            self.run_state = RunState::Paused;
            self.step();
        }

        if self.run_state == RunState::Running {
            ctx.request_repaint();
            self.step();

            let duration = self.target_frame_time - self.previous_frame_render_time;
            while (Instant::now() - start) < duration {
                self.step();
            }
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        _ = frame;

        let render_start = Instant::now();

        egui::Panel::top("top control panel").show_inside(ui, |ui| self.draw_header(ui));
        egui::CentralPanel::default().show_inside(ui, |ui| {
            egui_plot::Plot::new("results").show(ui, |plot_ui| {
                match self.view_mode {
                    ViewMode::Patterns => self.draw_pattern_bars(plot_ui),
                    ViewMode::Bits => {
                        self.draw_bits(plot_ui);
                    }
                };
            });
        });

        self.previous_frame_render_time = Instant::now() - render_start;
    }
}
