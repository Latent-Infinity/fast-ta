//! fast-ta command-line interface
//!
//! This binary provides a command-line interface for computing technical
//! analysis indicators on CSV data files.

pub mod error;

pub use error::{CliError, Result};

use clap::Parser;

/// Command-line arguments for fast-ta
#[derive(Parser, Debug)]
#[command(name = "fast-ta")]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input CSV file path
    #[arg(short, long)]
    input: Option<String>,

    /// Output CSV file path (defaults to stdout)
    #[arg(short, long)]
    output: Option<String>,

    /// Indicator to compute (e.g., sma, ema, rsi)
    #[arg(short = 'I', long)]
    indicator: Option<String>,

    /// Period for the indicator
    #[arg(short, long, default_value = "14")]
    period: usize,
}

fn main() {
    let args = Args::parse();

    // Placeholder - will be implemented in future phases
    println!("fast-ta CLI v{}", env!("CARGO_PKG_VERSION"));
    if let Some(input) = &args.input {
        println!("Input file: {input}");
    }
    if let Some(indicator) = &args.indicator {
        println!("Indicator: {indicator} (period: {})", args.period);
    }
}
