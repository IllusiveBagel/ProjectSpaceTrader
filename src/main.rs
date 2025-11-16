mod galaxy;

use std::env;
use std::fs::File;
use galaxy::Config;

fn parse_arg<T: std::str::FromStr>(args: &Vec<String>, key: &str) -> Option<T> {
    args.windows(2)
        .find(|w| w[0] == key)
        .and_then(|w| w[1].parse::<T>().ok())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut cfg = Config::default();

    // load JSON config if provided
    if let Some(conf_path) = parse_arg::<String>(&args, "--config") {
        match File::open(&conf_path) {
            Ok(f) => match serde_json::from_reader(f) {
                Ok(parsed) => cfg = parsed,
                Err(e) => {
                    eprintln!("Failed to parse JSON config '{}': {}", conf_path, e);
                    std::process::exit(1);
                }
            },
            Err(e) => {
                eprintln!("Failed to open config '{}': {}", conf_path, e);
                std::process::exit(1);
            }
        }
    }

    if let Some(s) = parse_arg::<u64>(&args, "--seed") {
        cfg.seed = s;
    }
    if let Some(m) = parse_arg::<i32>(&args, "--min") {
        cfg.min = m;
    }
    if let Some(m) = parse_arg::<i32>(&args, "--max") {
        cfg.max = m;
    }
    if let Some(md) = parse_arg::<f64>(&args, "--min-dist") {
        cfg.min_dist = md;
    }
    if let Some(md) = parse_arg::<f64>(&args, "--max-dist") {
        cfg.max_dist = Some(md);
    }
    if let Some(ma) = parse_arg::<usize>(&args, "--max-adjacent") {
        cfg.max_adjacent = ma;
    }
    if let Some(d) = parse_arg::<f64>(&args, "--density") {
        cfg.density = d;
    }
    if let Some(sm) = parse_arg::<f64>(&args, "--sample-multiplier") {
        cfg.sample_multiplier = sm;
    }
    if args.iter().any(|a| a == "--export-csv") {
        cfg.export_csv = true;
    }
    if let Some(a) = parse_arg::<usize>(&args, "--arms") {
        cfg.arms = a;
    }
    if let Some(s) = parse_arg::<f64>(&args, "--arm-spread") {
        cfg.arm_spread = s;
    }
    if let Some(t) = parse_arg::<f64>(&args, "--arm-twist") {
        cfg.arm_twist = t;
    }
    if let Some(s) = parse_arg::<f64>(&args, "--arm-strength") {
        cfg.arm_strength = s;
    }
    if let Some(s) = parse_arg::<f64>(&args, "--core-strength") {
        cfg.core_strength = s;
    }
    if let Some(r) = parse_arg::<f64>(&args, "--core-radius") {
        cfg.core_radius = r;
    }
    if let Some(f) = parse_arg::<f64>(&args, "--arm-falloff") {
        cfg.arm_falloff = f;
    }
    if let Some(sm) = parse_arg::<f64>(&args, "--sample-multiplier") {
        cfg.sample_multiplier = sm;
    }
    if args.iter().any(|a| a == "--export-csv") {
        cfg.export_csv = true;
    }
    // visualization flags
    if args.iter().any(|a| a == "--no-grid") {
        cfg.draw_grid = false;
    }
    if let Some(gs) = parse_arg::<usize>(&args, "--grid-step") {
        cfg.grid_step = gs;
    }
    if args.iter().any(|a| a == "--no-axes") {
        cfg.show_axes = false;
    }
    if let Some(r) = parse_arg::<f64>(&args, "--black-hole-radius") {
        cfg.black_hole_radius = r;
    }
    if let Some(w) = parse_arg::<f64>(&args, "--accretion-width") {
        cfg.accretion_width = w;
    }
    if let Some(r) = parse_arg::<f64>(&args, "--clear-radius") {
        cfg.clear_radius = r;
    }
    if args.iter().any(|a| a == "--orbital") {
        cfg.orbital = true;
    }
    if let Some(r) = parse_arg::<f64>(&args, "--arm-rotation") {
        cfg.arm_rotation = r;
    }
    if let Some(s) = parse_arg::<u32>(&args, "--scale") {
        cfg.scale = s;
    }
    if let Some(out) = args.windows(2).find(|w| w[0] == "--out").map(|w| w[1].clone()) {
        cfg.out = out;
    }
    if let Some(ts) = parse_arg::<usize>(&args, "--target-stars") {
        cfg.target_stars = Some(ts);
    }

    println!("Using config: {:?}", cfg);
    // generate galaxy
    let (grid, kept) = galaxy::generate_galaxy(&cfg);

    // TUI mode: if requested, render an ASCII view in the terminal and exit
    if args.iter().any(|a| a == "--tui") {
        if let Err(e) = galaxy::print_tui(&kept, &cfg) {
            eprintln!("Failed to render TUI: {}", e);
            std::process::exit(1);
        }
        return;
    }
    // Interactive mouse-driven TUI
    if args.iter().any(|a| a == "--tui-mouse") {
        // optional aspect ratio override for terminal character cells (pass 0 to auto)
        let tui_aspect = parse_arg::<f64>(&args, "--tui-aspect").unwrap_or(0.0f64);
        // braille rendering is the default; pass --no-braille to disable
        let use_braille = !args.iter().any(|a| a == "--no-braille");
        if let Err(e) = galaxy::interactive_tui(&kept, &cfg, tui_aspect, use_braille) {
            eprintln!("Failed to run interactive TUI: {}", e);
            std::process::exit(1);
        }
        return;
    }

    if let Err(e) = galaxy::export_image(&grid, &kept, &cfg) {
        eprintln!("Failed to save image: {}", e);
        std::process::exit(1);
    }

    println!("Placed {} stars. Output written to {}", kept.len(), cfg.out);
}
