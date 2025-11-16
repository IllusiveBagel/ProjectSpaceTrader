use image::{Rgb, RgbImage};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use noise::{NoiseFn, Perlin};

// small splitmix64 for deriving per-arm deterministic seeds
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct Config {
    pub seed: u64,
    pub min: i32,
    pub max: i32,
    pub min_dist: f64,
    pub max_dist: Option<f64>,
    pub max_adjacent: usize,
    pub density: f64,
    // sampling and export
    pub sample_multiplier: f64,
    pub export_csv: bool,
    // visualization: grid and black hole
    pub draw_grid: bool,
    pub grid_step: usize,
    pub show_axes: bool,
    pub black_hole_radius: f64,
    pub accretion_width: f64,
    // minimum clear radius around black hole where no stars are allowed
    pub clear_radius: f64,
    // spiral-specific parameters
    pub arms: usize,
    pub arm_spread: f64,
    pub arm_twist: f64,
    pub arm_strength: f64,
    pub core_strength: f64,
    pub core_radius: f64,
    pub arm_falloff: f64,
    pub scale: u32,
    pub out: String,
    // orbital
    pub orbital: bool,
    pub arm_rotation: f64,
    // number of sampling attempts per arm per radial shell when streaming outward
    pub samples_per_shell: usize,
    // optional cap on number of final stars; if Some(n) we'll stop after placing n stars
    pub target_stars: Option<usize>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            seed: 42,
            min: -50,
            max: 50,
            min_dist: 2.0,
            max_dist: Some(12.0),
            max_adjacent: 3,
            // increase density and sampling so more candidates are considered
            density: 0.06,
            sample_multiplier: 8.0,
            export_csv: false,
            draw_grid: true,
            grid_step: 10,
            show_axes: true,
            black_hole_radius: 2.0,
            accretion_width: 6.0,
            clear_radius: 4.0,
            // spiral parameters
            arms: 3,
            arm_spread: 0.5,
            arm_twist: 0.25,
            arm_strength: 1.0,
            core_strength: 2.5,
            core_radius: 6.0,
            arm_falloff: 0.02,
            scale: 6,
            out: "galaxy.png".to_string(),
            orbital: true,
            arm_rotation: 0.6,
            target_stars: None,
            samples_per_shell: 3,
        }
    }
}

// (removed unused helper wrap_angle to silence a compiler warning)

pub fn generate_galaxy(cfg: &Config) -> (Vec<Vec<bool>>, Vec<(f64, f64)>) {
    // New approach: sample many continuous candidates weighted by spiral/core density,
    // then perform distance-based filtering (keep highest-density candidates first)
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);

    let size = (cfg.max - cfg.min + 1) as usize;
    let mut grid = vec![vec![false; size]; size];

    let width = (cfg.max - cfg.min) as f64;
    let height = (cfg.max - cfg.min) as f64;
    let area = width * height;

    // helper to compute raw combined density (core + arm) without cfg.density scaling
    let max_combined = cfg.core_strength + cfg.arm_strength;
    let cx = (cfg.min + cfg.max) as f64 / 2.0;
    let cy = (cfg.min + cfg.max) as f64 / 2.0;

    let sample_count = ((area.max(1.0)) * cfg.sample_multiplier) as usize;
    // avoid reserving an enormous amount of memory when sample_multiplier is large;
    // cap the initial reserve to a sane upper bound (will grow if needed).
    let reserve_cap = 500_000usize;
    let reserve = sample_count.min(reserve_cap);
    let mut candidates: Vec<(f64, f64, f64)> = Vec::with_capacity(reserve);

    // spatial grid for fast neighbor checking (create early so orbital mode can insert directly)
    let cell_size = cfg.min_dist.max(0.0001);
    let grid_w = ((cfg.max - cfg.min) as f64 / cell_size).ceil() as usize + 1;
    let grid_h = grid_w;
    let mut buckets: Vec<Vec<(f64, f64)>> = vec![Vec::new(); grid_w * grid_h];
    let mut kept: Vec<(f64, f64)> = Vec::new();
    let mut orbital_direct = false;

    // Two sampling modes: orbital parametric sampling (Archimedean spiral arms) or random weighted sampling
    let max_radius = (((cfg.max - cfg.min) as f64) * 0.5).hypot(((cfg.max - cfg.min) as f64) * 0.5);

    // compute an effective clear radius that ensures no stars are placed inside
    // the black hole or its accretion disk. We use this for sampling and
    // rasterization checks so visible black hole area is always empty.
    let bh_outer = cfg.black_hole_radius + cfg.accretion_width;
    let effective_clear = cfg.clear_radius.max(bh_outer + 0.5);
    let effective_clear_sq = effective_clear * effective_clear;
    let min_dist_sq = cfg.min_dist * cfg.min_dist;

    if cfg.orbital {
        // Archimedean spiral: r = s * theta, theta >= 0
        let s = cfg.arm_twist.max(0.001); // radial growth per radian
        let theta_max = (max_radius / s).max(1.0);

        // determine number of samples per arm
        let total_per_arm = ((theta_max * cfg.density * cfg.sample_multiplier) as usize).max(10);
        let arm_phase = (cfg.seed as f64).to_bits() as f64 % (2.0 * std::f64::consts::PI);

    // core (bulge) sampling — keep this limited so the bulge doesn't dominate.
    // scale down core sampling so outer disk and arms receive more candidates.
    let core_samples_full = ((cfg.density * cfg.sample_multiplier * 2.0) * (cfg.core_radius * cfg.core_radius * std::f64::consts::PI)).ceil() as usize;
    let core_samples = (core_samples_full as f64 * 0.25).ceil() as usize; // only 25% of naive core samples
    let core_direct_keep_prob = 0.08f64; // keep only a small fraction of direct core samples to avoid central crowding
        for _ in 0..core_samples {
            let rr = rng.sample::<f64, _>(rand_distr::Normal::new(0.0, cfg.core_radius).unwrap()).abs();
            let ang = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
            let rx = cx + rr * ang.cos();
            let ry = cy + rr * ang.sin();
            let combined = cfg.core_strength * (-(rr * rr) / (2.0 * cfg.core_radius * cfg.core_radius)).exp();
            // check eviction against effective_clear and neighbor buckets
            let dx_bh = rx - cx;
            let dy_bh = ry - cy;
            let dist_bh_sq = dx_bh * dx_bh + dy_bh * dy_bh;
            if dist_bh_sq <= effective_clear_sq { continue; }

            // only accept a small fraction directly into kept to avoid core dominance
            if rng.gen_range(0.0..1.0) > core_direct_keep_prob {
                // push as a candidate instead so the main filtering can decide placement
                candidates.push((rx, ry, combined));
                continue;
            }

            let bx = ((rx - cfg.min as f64) / cell_size).floor() as isize;
            let by = ((ry - cfg.min as f64) / cell_size).floor() as isize;
            if bx < 0 || by < 0 { continue; }
            let bx = bx as usize;
            let by = by as usize;
            if bx >= grid_w || by >= grid_h { continue; }
            let mut too_close = false;
            let min_bx = bx.saturating_sub(1);
            let min_by = by.saturating_sub(1);
            let max_bx = (bx + 1).min(grid_w - 1);
            let max_by = (by + 1).min(grid_h - 1);
            for yy in min_by..=max_by {
                for xx in min_bx..=max_bx {
                    let idx = yy * grid_w + xx;
                    for &(kx, ky) in &buckets[idx] {
                        let dx = kx - rx;
                        let dy = ky - ry;
                        if (dx * dx + dy * dy) < min_dist_sq {
                            too_close = true;
                            break;
                        }
                    }
                    if too_close { break; }
                }
                if too_close { break; }
            }
            if too_close { continue; }
            kept.push((rx, ry));
            // console progress: print every 100 stars and the first 20 for visibility
            let placed = kept.len();
            if placed <= 20 || placed % 100 == 0 {
                println!("Added star #{} at ({:.2},{:.2})", placed, rx, ry);
            }
            let idx = by * grid_w + bx;
            buckets[idx].push((rx, ry));
            if let Some(limit) = cfg.target_stars {
                if kept.len() >= limit { orbital_direct = true; break; }
            }
        }

        // Streaming outward generation: if a target number of stars is requested,
        // generate candidates shell-by-shell from the black hole outward and try
        // to insert them immediately. This ensures the galaxy grows outward and
        // fills the arms instead of forming a dense central belt.
        if let Some(limit) = cfg.target_stars {
            let arm_count = cfg.arms.max(1);
            // per-arm deterministic RNGs and Perlin instances
            let mut arm_rngs: Vec<ChaCha8Rng> = (0..arm_count).map(|i| {
                let seed_i = splitmix64(cfg.seed.wrapping_add(i as u64));
                ChaCha8Rng::seed_from_u64(seed_i)
            }).collect();
            let perlins: Vec<Perlin> = (0..arm_count).map(|i| {
                let seed_i = splitmix64(cfg.seed.wrapping_add(i as u64));
                Perlin::new((seed_i & 0xffff_ffff) as u32)
            }).collect();

            let r_start = effective_clear + cfg.min_dist;
            let r_step = (cfg.min_dist * 0.9).max(0.25);
            let mut r = r_start;
            // number of samples per arm per shell controlled by cfg.samples_per_shell
            'outer: while r <= max_radius {
                for i in 0..arm_count {
                    if kept.len() >= limit { break 'outer; }
                    let arm_base = (i as f64) * 2.0 * std::f64::consts::PI / (arm_count as f64);
                    let perlin = &perlins[i];
                    let rng_arm = &mut arm_rngs[i];

                    // map radius to theta along the Archimedean spiral
                    let theta = (r / s).max(0.0);
                    let t = (theta / theta_max).clamp(0.0, 1.0);

                    // angular wiggle via a small FBM (2 octaves)
                    let mut f = 0.8 / (1.0 + theta.max(1.0));
                    let mut amp = 1.0;
                    let mut n_ang = 0.0;
                    let mut total_amp = 0.0;
                    for _oct in 0..2 {
                        n_ang += amp * perlin.get([theta * f, (i as f64) * 0.7]);
                        total_amp += amp;
                        amp *= 0.5;
                        f *= 2.0;
                    }
                    n_ang /= total_amp.max(1e-9);
                    let wiggle_amp = 1.2 * (1.0 - t).max(0.0);
                    let angle_noise = n_ang * wiggle_amp;
                    let rot = cfg.arm_rotation * (r / max_radius);
                    let angle = arm_phase + arm_base + theta + rot + angle_noise;

                    // radial wiggle (FBM)
                    let mut f2 = 0.8 * 0.7;
                    let mut amp2 = 1.0;
                    let mut n_rad = 0.0;
                    let mut total2 = 0.0;
                    for _oct in 0..2 {
                        n_rad += amp2 * perlin.get([theta * f2 + 12.34, (i as f64) * 0.9 + 4.56]);
                        total2 += amp2;
                        amp2 *= 0.5;
                        f2 *= 2.1;
                    }
                    n_rad /= total2.max(1e-9);
                    let radial_amp = cfg.arm_spread.max(0.01) * (0.5 + 0.9 * (1.0 - t));
                    let offset_r = n_rad * radial_amp * (1.0 + r / (max_radius + 1.0));

                for _s in 0..cfg.samples_per_shell {
                let micro = rng_arm.sample::<f64, _>(rand_distr::Normal::new(0.0, 0.08).unwrap());
                // use a small normal angular jitter to keep samples close to the arm center
                let ang_jitter = rng_arm.sample::<f64, _>(rand_distr::Normal::new(0.0, 0.12).unwrap());
                let angle_s = angle + ang_jitter;
                let rx = cx + (r + offset_r + micro) * angle_s.cos();
                let ry = cy + (r + offset_r + micro) * angle_s.sin();

                    // skip if inside effective clear radius
                    let dx_bh = rx - cx;
                    let dy_bh = ry - cy;
                    let dist_bh_sq = dx_bh * dx_bh + dy_bh * dy_bh;
                    if dist_bh_sq <= effective_clear_sq { continue; }

                        // quick bounds check
                        let bx = ((rx - cfg.min as f64) / cell_size).floor() as isize;
                        let by = ((ry - cfg.min as f64) / cell_size).floor() as isize;
                        if bx < 0 || by < 0 { continue; }
                        let bxu = bx as usize;
                        let byu = by as usize;
                        if bxu >= grid_w || byu >= grid_h { continue; }

                    // neighbor check using min_dist enforcement
                    let mut too_close = false;
                    let min_bx = bxu.saturating_sub(1);
                    let min_by = byu.saturating_sub(1);
                    let max_bx = (bxu + 1).min(grid_w - 1);
                    let max_by = (byu + 1).min(grid_h - 1);
                    for yy in min_by..=max_by {
                        for xx in min_bx..=max_bx {
                            let idx = yy * grid_w + xx;
                            for &(kx, ky) in &buckets[idx] {
                                let dx = kx - rx;
                                let dy = ky - ry;
                                if (dx * dx + dy * dy) < min_dist_sq {
                                    too_close = true;
                                    break;
                                }
                            }
                            if too_close { break; }
                        }
                        if too_close { break; }
                    }
                        if too_close { continue; }

                        // accept directly into kept
                        kept.push((rx, ry));
                        let placed = kept.len();
                        if placed <= 20 || placed % 100 == 0 {
                            println!("Added star #{} at ({:.2},{:.2})", placed, rx, ry);
                        }
                        let idx = byu * grid_w + bxu;
                        buckets[idx].push((rx, ry));
                    }
                }
                r += r_step;
            }
            // If we reached the target, mark orbital_direct to skip the candidate pipeline
            if kept.len() >= limit {
                orbital_direct = true;
            }
        }

        // Parallel per-arm candidate generation: heavy Perlin/FBM math is parallelized per arm.
        let arm_count = cfg.arms.max(1);
    // generate per-arm candidates sequentially (avoids high parallel peak memory)
    let per_arm_results: Vec<Vec<(f64,f64,f64)>> = (0..arm_count).map(|i| {
            let arm_base = (i as f64) * 2.0 * std::f64::consts::PI / (arm_count as f64);
            // per-arm deterministic RNG seed
            let seed_i = splitmix64(cfg.seed.wrapping_add(i as u64));
            let mut rng_arm = ChaCha8Rng::seed_from_u64(seed_i);
            let perlin = Perlin::new((seed_i & 0xffff_ffff) as u32);

            // compute per-arm sample count and stepping
            let mut per_arm = ((theta_max * cfg.sample_multiplier * 2.0).ceil() as usize).max(total_per_arm);
            let cap = 100_000usize;
            if per_arm > cap { per_arm = cap; }
            let freq = 0.8 / (1.0 + theta_max.max(1.0));
            let base_step = (theta_max / (per_arm.max(16) as f64)).max(0.02);

            let mut local_cands: Vec<(f64,f64,f64)> = Vec::new();
            let mut theta = rng_arm.gen_range(0.0..base_step);
            while theta < theta_max {
                let t = (theta / theta_max).clamp(0.0, 1.0);
                let r = s * theta;

                // angular wiggle via a small FBM (2 octaves)
                let mut f = freq;
                let mut amp = 1.0;
                let mut n_ang = 0.0;
                let mut total_amp = 0.0;
                for _oct in 0..2 {
                    n_ang += amp * perlin.get([theta * f, (i as f64) * 0.7]);
                    total_amp += amp;
                    amp *= 0.5;
                    f *= 2.0;
                }
                n_ang /= total_amp.max(1e-9);
                let wiggle_amp = 1.2 * (1.0 - t).max(0.0);
                let angle_noise = n_ang * wiggle_amp;
                let rot = cfg.arm_rotation * (r / max_radius);
                let angle = arm_phase + arm_base + theta + rot + angle_noise;

                // radial wiggle (FBM)
                let mut f2 = freq * 0.7;
                let mut amp2 = 1.0;
                let mut n_rad = 0.0;
                let mut total2 = 0.0;
                for _oct in 0..2 {
                    n_rad += amp2 * perlin.get([theta * f2 + 12.34, (i as f64) * 0.9 + 4.56]);
                    total2 += amp2;
                    amp2 *= 0.5;
                    f2 *= 2.1;
                }
                n_rad /= total2.max(1e-9);
                let radial_amp = cfg.arm_spread.max(0.01) * (0.5 + 0.9 * (1.0 - t));
                let offset_r = n_rad * radial_amp * (1.0 + r / (max_radius + 1.0));

                let micro = rng_arm.sample::<f64, _>(rand_distr::Normal::new(0.0, 0.12).unwrap());
                let rx = cx + (r + offset_r + micro) * angle.cos();
                let ry = cy + (r + offset_r + micro) * angle.sin();

                // skip if inside effective clear radius
                let dx_bh = rx - cx;
                let dy_bh = ry - cy;
                let dist_bh_sq = dx_bh * dx_bh + dy_bh * dy_bh;
                if dist_bh_sq <= effective_clear_sq { theta += base_step; continue; }

                // density modulation (small FBM)
                let mut dens_fb = 0.0;
                let mut da = 1.0;
                let mut ff = freq * 1.1;
                let mut tot = 0.0;
                for _oct in 0..2 {
                    dens_fb += da * perlin.get([theta * ff + 77.7, (i as f64) * 1.23]);
                    tot += da;
                    da *= 0.5;
                    ff *= 1.9;
                }
                dens_fb /= tot.max(1e-9);
                let dens_mod = ((dens_fb * 0.5) + 0.5).clamp(0.0, 1.0);

                let spread = cfg.arm_spread.max(0.01);
                let arm_alignment = (-(offset_r * offset_r) / (2.0 * spread * spread)).exp();
                let radial_decay = (-(r / ((max_radius + 1.0) * cfg.arm_falloff)).abs()).exp();
                let combined = cfg.arm_strength * arm_alignment * radial_decay + cfg.core_strength * (-(r * r) / (2.0 * cfg.core_radius * cfg.core_radius)).exp() * 0.1;
                // favor placements farther from the center so stars spread across the disk
                let radial_frac = (r / max_radius).clamp(0.0, 1.0);
                // radial preference during sampling: strongly favor outer disk (non-linear)
                // this reduces overpopulation of the very center and encourages filling the arms
                let radial_weight = 0.05 + 0.95 * radial_frac.powf(1.3);
                // amplify density modulation along arms so arm-local peaks are more likely
                // increase contrast: reduce baseline and amplify peaks so arms are clearer
                let density_prob = (cfg.density * (0.25 + 1.6 * dens_mod) * combined / max_combined * radial_weight).clamp(0.0, 1.0);
                if rng_arm.gen_range(0.0..1.0) <= density_prob {
                    local_cands.push((rx, ry, combined));
                    // safety cap per arm to avoid unbounded memory use
                    const PER_ARM_MAX: usize = 50_000;
                    if local_cands.len() >= PER_ARM_MAX { break; }
                }

                // advance theta with modulated step
                let step_noise = perlin.get([theta * (freq * 0.7), (i as f64) * 2.2]);
                let step_factor = (0.6 + 0.8 * dens_mod) * (1.0 + 0.3 * step_noise);
                let rand_factor: f64 = 0.7 + rng_arm.gen_range(0.0..0.8);
                theta += base_step * step_factor.max(0.2) * rand_factor;
            }
            local_cands
        }).collect();

        // flatten per-arm results into candidates, but cap the total candidate pool
        // to avoid unbounded memory when sample_multiplier is large. We keep the
        // top-scoring candidates by a simple truncate-after-sort approach.
        const MAX_CANDIDATES: usize = 300_000;
        for v in per_arm_results {
            for c in v {
                candidates.push(c);
            }
            if candidates.len() > MAX_CANDIDATES {
                // compute score: combined * radial_weight
                let cx_loc = cx;
                let cy_loc = cy;
                let max_r_loc = max_radius;
                candidates.sort_by(|a, b| {
                    let ra = ((a.0 - cx_loc) * (a.0 - cx_loc) + (a.1 - cy_loc) * (a.1 - cy_loc)).sqrt();
                    let rb = ((b.0 - cx_loc) * (b.0 - cx_loc) + (b.1 - cy_loc) * (b.1 - cy_loc)).sqrt();
                    let fra = (ra / max_r_loc).clamp(0.0, 1.0);
                    let frb = (rb / max_r_loc).clamp(0.0, 1.0);
                    // stronger outer bias for truncation: prefer farther candidates
                    let wa = 0.05 + 0.95 * fra.powf(1.5);
                    let wb = 0.05 + 0.95 * frb.powf(1.5);
                    let sa = a.2 * wa;
                    let sb = b.2 * wb;
                    sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                });
                candidates.truncate(MAX_CANDIDATES);
            }
        }

        // core bulge: generate core candidates (serial, but reduced so core doesn't dominate)
        let core_seed = splitmix64(cfg.seed.wrapping_add(0x12345));
        let mut core_rng = ChaCha8Rng::seed_from_u64(core_seed);
    let core_keep_prob = 0.04; // keep a smaller fraction of core candidates directly to avoid central crowding
        for _ in 0..core_samples {
            let rr = core_rng.sample::<f64, _>(rand_distr::Normal::new(0.0, cfg.core_radius).unwrap()).abs();
            let ang = core_rng.gen_range(0.0..2.0 * std::f64::consts::PI);
            let rx = cx + rr * ang.cos();
            let ry = cy + rr * ang.sin();
            let combined = cfg.core_strength * (-(rr * rr) / (2.0 * cfg.core_radius * cfg.core_radius)).exp();
            let dx_bh = rx - cx;
            let dy_bh = ry - cy;
            let dist_bh_sq = dx_bh * dx_bh + dy_bh * dy_bh;
            if dist_bh_sq <= effective_clear_sq { continue; }
            if core_rng.gen_range(0.0..1.0) <= core_keep_prob {
                candidates.push((rx, ry, combined * 0.5));
            }
        }
    }

    // If orbital_direct is false, we still need to process the generated candidates
    if !orbital_direct {
        // sort candidates by a score that favors high combined strength but also
        // prefers points farther from the center (so arms fill before the core)
        candidates.sort_by(|a, b| {
            let ra = ((a.0 - cx) * (a.0 - cx) + (a.1 - cy) * (a.1 - cy)).sqrt();
            let rb = ((b.0 - cx) * (b.0 - cx) + (b.1 - cy) * (b.1 - cy)).sqrt();
            let fra = (ra / max_radius).clamp(0.0, 1.0);
            let frb = (rb / max_radius).clamp(0.0, 1.0);
                // radial bias for sorting: favor outer candidates but not too extremely
                let wa = 0.25 + 0.75 * fra; // 0.25..1.0
                let wb = 0.25 + 0.75 * frb;
            let sa = a.2 * wa;
            let sb = b.2 * wb;
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });

    let mut reached_target = false;
    let mut rejected_candidates: Vec<(f64, f64, f64)> = Vec::new();
    for (x, y, combined) in candidates {
            let bx = ((x - cfg.min as f64) / cell_size).floor() as isize;
            let by = ((y - cfg.min as f64) / cell_size).floor() as isize;
            if bx < 0 || by < 0 { continue; }
            let bx = bx as usize;
            let by = by as usize;
            if bx >= grid_w || by >= grid_h { continue; }

            let mut too_close = false;
            // check neighboring buckets
            let min_bx = bx.saturating_sub(1);
            let min_by = by.saturating_sub(1);
            let max_bx = (bx + 1).min(grid_w - 1);
            let max_by = (by + 1).min(grid_h - 1);
            for yy in min_by..=max_by {
                for xx in min_bx..=max_bx {
                    let idx = yy * grid_w + xx;
                    for &(kx, ky) in &buckets[idx] {
                        let dx = kx - x;
                        let dy = ky - y;
                        if (dx * dx + dy * dy) < min_dist_sq {
                            too_close = true;
                            break;
                        }
                    }
                    if too_close { break; }
                }
                if too_close { break; }
            }
            if too_close {
                // keep rejected candidates for a deterministic second-pass filler
                rejected_candidates.push((x, y, combined));
                continue;
            }

            // accept
            kept.push((x, y));
            let placed = kept.len();
            if placed <= 20 || placed % 100 == 0 {
                println!("Added star #{} at ({:.2},{:.2})", placed, x, y);
            }
            let idx = by * grid_w + bx;
            buckets[idx].push((x, y));
            if let Some(limit) = cfg.target_stars {
                if kept.len() >= limit {
                    reached_target = true;
                    break;
                }
            }
        }
        if reached_target {
            // if we stopped early, we may want to discard remaining candidates
        }

            // Second-pass: if we didn't reach the target, try placing rejected
            // candidates from the outside in (favor filling arms), using the
            // same min_dist enforcement. This helps spread stars away from
            // the core without relaxing spacing.
            if !reached_target {
                // sort rejected by radial distance (farthest first)
                rejected_candidates.sort_by(|a, b| {
                    let ra = ((a.0 - cx)*(a.0 - cx) + (a.1 - cy)*(a.1 - cy)).partial_cmp(&((b.0 - cx)*(b.0 - cx) + (b.1 - cy)*(b.1 - cy))).unwrap_or(std::cmp::Ordering::Equal);
                    ra.reverse()
                });

                for (x, y, _combined) in &rejected_candidates {
                    if let Some(limit) = cfg.target_stars {
                        if kept.len() >= limit { break; }
                    }
                    let bx = ((*x - cfg.min as f64) / cell_size).floor() as isize;
                    let by = ((*y - cfg.min as f64) / cell_size).floor() as isize;
                    if bx < 0 || by < 0 { continue; }
                    let bx = bx as usize;
                    let by = by as usize;
                    if bx >= grid_w || by >= grid_h { continue; }
                    let mut too_close = false;
                    let min_bx = bx.saturating_sub(1);
                    let min_by = by.saturating_sub(1);
                    let max_bx = (bx + 1).min(grid_w - 1);
                    let max_by = (by + 1).min(grid_h - 1);
                    for yy in min_by..=max_by {
                        for xx in min_bx..=max_bx {
                            let idx = yy * grid_w + xx;
                            for &(kx, ky) in &buckets[idx] {
                                let dx = kx - x;
                                let dy = ky - y;
                                if (dx * dx + dy * dy) < min_dist_sq {
                                    too_close = true;
                                    break;
                                }
                            }
                            if too_close { break; }
                        }
                        if too_close { break; }
                    }
                    if too_close { continue; }
                    kept.push((*x, *y));
                    let placed = kept.len();
                    if placed <= 20 || placed % 100 == 0 {
                        println!("Added star #{} at ({:.2},{:.2})", placed, *x, *y);
                    }
                    let idx = by * grid_w + bx;
                    buckets[idx].push((*x, *y));
                }
                // If we still haven't reached the target, do a relaxed-spacing
                // filler here (we're inside the scope where `rejected_candidates`,
                // `buckets`, `cell_size`, etc. are visible).
                if let Some(limit) = cfg.target_stars {
                    if kept.len() < limit {
                        let mut placed_relaxed = 0usize;
                        let relax_steps = [0.9, 0.8, 0.7, 0.6, 0.5];
                        for &factor in &relax_steps {
                            if kept.len() >= limit { break; }
                            let relaxed_min_sq = (cfg.min_dist * factor) * (cfg.min_dist * factor);
                            for (rx, ry, _combined) in &rejected_candidates {
                                if kept.len() >= limit { break; }
                                let bx = ((rx - cfg.min as f64) / cell_size).floor() as isize;
                                let by = ((ry - cfg.min as f64) / cell_size).floor() as isize;
                                if bx < 0 || by < 0 { continue; }
                                let bx = bx as usize;
                                let by = by as usize;
                                if bx >= grid_w || by >= grid_h { continue; }
                                let mut too_close = false;
                                let min_bx = bx.saturating_sub(1);
                                let min_by = by.saturating_sub(1);
                                let max_bx = (bx + 1).min(grid_w - 1);
                                let max_by = (by + 1).min(grid_h - 1);
                                for yy in min_by..=max_by {
                                    for xx in min_bx..=max_bx {
                                        let idx = yy * grid_w + xx;
                                        for &(kx, ky) in &buckets[idx] {
                                            let dx = kx - rx;
                                            let dy = ky - ry;
                                            if (dx * dx + dy * dy) < relaxed_min_sq {
                                                too_close = true;
                                                break;
                                            }
                                        }
                                        if too_close { break; }
                                    }
                                    if too_close { break; }
                                }
                                if too_close { continue; }
                                kept.push((*rx, *ry));
                                let idx = by * grid_w + bx;
                                buckets[idx].push((*rx, *ry));
                                placed_relaxed += 1;
                            }
                        }
                        if placed_relaxed > 0 {
                            println!("Relaxed pass placed {} extra stars (kept total {}).", placed_relaxed, kept.len());
                        }
                    }
                }
            }
    }

    // rasterize kept points to boolean grid (for compatibility / indexing),
    // but we also return the continuous kept list so export can draw subpixel stars.
    for &(x, y) in &kept {
        // map world coordinate x (which ranges cfg.min..cfg.max) to grid index 0..size-1
        let ix = (x.round() as isize) - (cfg.min as isize);
        let iy = (y.round() as isize) - (cfg.min as isize);
        if ix >= 0 && iy >= 0 {
            let ix = ix as usize;
            let iy = iy as usize;
            if iy < size && ix < size {
                // compute world-space center of this cell
                let wx = cfg.min as f64 + ix as f64;
                let wy = cfg.min as f64 + iy as f64;
                let dx = wx - cx;
                let dy = wy - cy;
                let rcell = (dx * dx + dy * dy).sqrt();
                // skip placing a star if the rasterized cell lies inside the effective clear radius
                if rcell <= effective_clear {
                    continue;
                }
                grid[iy][ix] = true;
            }
        }
    }

    // optionally write CSV of star coords (continuous)
    if cfg.export_csv {
        let csv_path = std::path::Path::new(&cfg.out).with_extension("csv");
        if let Ok(mut w) = std::fs::File::create(csv_path) {
            use std::io::Write;
            let _ = writeln!(w, "x,y");
            for &(x, y) in &kept {
                let _ = writeln!(w, "{:.4},{:.4}", x, y);
            }
        }
    }

    (grid, kept)
}

pub fn export_image(grid: &Vec<Vec<bool>>, kept: &Vec<(f64,f64)>, cfg: &Config) -> Result<(), Box<dyn std::error::Error>> {
    let height = grid.len();
    let width = grid[0].len();
    let img_w = (width as u32) * cfg.scale;
    let img_h = (height as u32) * cfg.scale;

    let mut img = RgbImage::from_pixel(img_w, img_h, Rgb([0u8, 0u8, 0u8]));

    // draw grid and black hole first (so stars overlay them)
    let center_cell_x = (0i32 - cfg.min) as i32; // may be negative if 0 out of range
    let center_cell_y = (0i32 - cfg.min) as i32;
    let center_px = if center_cell_x >= 0 { (center_cell_x as u32) * cfg.scale + cfg.scale / 2 } else { img_w / 2 };
    let center_py = if center_cell_y >= 0 { (center_cell_y as u32) * cfg.scale + cfg.scale / 2 } else { img_h / 2 };

    if cfg.draw_grid {
        let grid_color = Rgb([40u8, 40u8, 40u8]);
        let axis_color = Rgb([0u8, 180u8, 0u8]);

        // vertical grid lines every grid_step cells
        if cfg.grid_step > 0 {
            let step = cfg.grid_step as i32;
            for gx in (cfg.min..=cfg.max).step_by(step as usize) {
                let ix = (gx - cfg.min) as isize;
                if ix < 0 { continue; }
                let px = (ix as u32) * cfg.scale;
                for y in 0..img_h {
                    img.put_pixel(px.min(img_w - 1), y, grid_color);
                }
            }
            // horizontal grid lines
            for gy in (cfg.min..=cfg.max).step_by(step as usize) {
                let iy = (gy - cfg.min) as isize;
                if iy < 0 { continue; }
                let py = (iy as u32) * cfg.scale;
                for x in 0..img_w {
                    img.put_pixel(x, py.min(img_h - 1), grid_color);
                }
            }
        }

        // (relaxed third-pass is executed earlier inside the candidate placement
        // block to avoid scope issues) -- nothing to do here.

        // draw axes at center (0,0)
        if cfg.show_axes {
            // vertical axis
            if center_px < img_w {
                for y in 0..img_h {
                    let _ = img.put_pixel(center_px, y, axis_color);
                    if center_px > 0 { let _ = img.put_pixel(center_px - 1, y, axis_color); }
                }
            }
            // horizontal axis
            if center_py < img_h {
                for x in 0..img_w {
                    let _ = img.put_pixel(x, center_py, axis_color);
                    if center_py > 0 { let _ = img.put_pixel(x, center_py - 1, axis_color); }
                }
            }
        }
    }

    // draw black hole with accretion disk at image center
    {
        let bh_inner = cfg.black_hole_radius * cfg.scale as f64;
        let bh_outer = bh_inner + cfg.accretion_width * cfg.scale as f64;
        let bh_inner_sq = bh_inner * bh_inner;
        let bh_outer_sq = bh_outer * bh_outer;
        for yy in 0..img_h {
            for xx in 0..img_w {
                let dx = (xx as i32 - center_px as i32) as f64;
                let dy = (yy as i32 - center_py as i32) as f64;
                let d2 = dx * dx + dy * dy;
                if d2 <= bh_inner_sq {
                    // event horizon -> black
                    img.put_pixel(xx, yy, Rgb([0u8, 0u8, 0u8]));
                } else if d2 <= bh_outer_sq {
                    // accretion disk: orange-yellow gradient
                    let d = d2.sqrt();
                    let t = ((d - bh_inner) / (bh_outer - bh_inner)).clamp(0.0, 1.0);
                    let r = (255.0 * (1.0 - t) + 180.0 * t) as u8;
                    let g = (120.0 * (1.0 - t) + 80.0 * t) as u8;
                    let b = (40.0 * (1.0 - t) + 20.0 * t) as u8;
                    img.put_pixel(xx, yy, Rgb([r, g, b]));
                }
            }
        }
    }

    // draw stars directly from continuous kept positions to avoid integer rounding artifacts
    for &(x, y) in kept {
        // map world coords to pixel-space (float)
        let fx = (x - cfg.min as f64) * cfg.scale as f64;
        let fy = (y - cfg.min as f64) * cfg.scale as f64;

        // star radius in pixels (small); vary slightly with distance from center for depth
    let dx_c = x - ((cfg.min + cfg.max) as f64 / 2.0);
    let dy_c = y - ((cfg.min + cfg.max) as f64 / 2.0);
    let dist = (dx_c * dx_c + dy_c * dy_c).sqrt();
    let max_r = ((cfg.scale as f64) * 1.2).max(1.0);
    // Make inner stars smaller to avoid the bright central blob. Inner radius is
    // reduced and outer stars get slightly larger to emphasize arms.
    let r = (max_r * (dist / ((cfg.max - cfg.min) as f64 * 0.5)).clamp(0.1, 1.0)).max(0.5);

        let r_i = r.ceil() as i32;
        let min_x = fx as i32 - r_i;
        let max_x = fx as i32 + r_i;
        let min_y = fy as i32 - r_i;
        let max_y = fy as i32 + r_i;

        // deterministic color/brightness variation per-star using a simple hash
        let hue_seed = ((x * 12.9898 + y * 78.233).sin() * 43758.5453).abs();
        let bright = (0.75 + 0.5 * (hue_seed.fract())) .clamp(0.0, 1.0);
        let base_r = (255.0 * bright) as u8;
        let base_g = (230.0 * bright) as u8;
        let base_b = (200.0 * bright) as u8;

        for yy in min_y..=max_y {
            if yy < 0 || yy as u32 >= img_h { continue; }
            for xx in min_x..=max_x {
                if xx < 0 || xx as u32 >= img_w { continue; }
                let dx = (xx as f64 + 0.5) - fx;
                let dy = (yy as f64 + 0.5) - fy;
                if dx * dx + dy * dy <= r * r {
                    img.put_pixel(xx as u32, yy as u32, Rgb([base_r, base_g, base_b]));
                }
            }
        }
    }

    let path = std::path::Path::new(&cfg.out);
    img.save(path)?;
    Ok(())
}

/// Render a coarse ASCII/TUI view of the galaxy to the terminal.
pub fn print_tui(kept: &Vec<(f64,f64)>, cfg: &Config) -> Result<(), Box<dyn std::error::Error>> {
    use crossterm::{execute, cursor, terminal::{self, ClearType}};
    use std::io::{stdout, Write};

    let (term_w, term_h) = terminal::size()?;
    let cols = term_w.max(10) as usize;
    // reserve two rows for info
    let rows = term_h.saturating_sub(2).max(10) as usize;

    // Create a coarse buffer
    let mut buf: Vec<Vec<u32>> = vec![vec![0u32; cols]; rows];

    let world_w = (cfg.max - cfg.min + 1) as f64;
    let world_h = (cfg.max - cfg.min + 1) as f64;

    for &(x, y) in kept.iter() {
        // map world coords to terminal grid
        let fx = ((x - cfg.min as f64) / world_w) * (cols as f64);
        let fy = ((y - cfg.min as f64) / world_h) * (rows as f64);
        let cx = fx.floor() as isize;
        let cy = fy.floor() as isize;
        if cx >= 0 && (cx as usize) < cols && cy >= 0 && (cy as usize) < rows {
            buf[cy as usize][cx as usize] = buf[cy as usize][cx as usize].saturating_add(1);
        }
    }

    // clear screen and print
    execute!(stdout(), terminal::Clear(ClearType::All), cursor::MoveTo(0,0))?;
    let mut out = String::new();
    for row in 0..rows {
        out.clear();
        for col in 0..cols {
            let v = buf[row][col];
            let ch = if v == 0 {
                ' '
            } else if v < 3 {
                '.'
            } else if v < 8 {
                '*'
            } else {
                '@'
            };
            out.push(ch);
        }
        out.push('\n');
        print!("{}", out);
    }
    // info line
    let info = format!("Stars: {}  region: {}..{}  samples_per_shell: {}", kept.len(), cfg.min, cfg.max, cfg.samples_per_shell);
    println!("{}", info);
    stdout().flush()?;
    Ok(())
}

/// Interactive TUI with mouse pan/zoom. Use left-button drag to pan, mouse wheel to zoom,
/// arrow keys to pan, +/- to zoom, and 'q' to quit.
pub fn interactive_tui(kept: &Vec<(f64,f64)>, cfg: &Config, char_aspect: f64, use_braille: bool) -> Result<(), Box<dyn std::error::Error>> {
    use crossterm::event::{self, Event, KeyCode, MouseButton, MouseEventKind};
    use crossterm::{execute, terminal, cursor};
    use std::io::stdout;
    use std::time::Duration;

    let mut stdout = stdout();
    terminal::enable_raw_mode()?;
    execute!(stdout, terminal::EnterAlternateScreen, cursor::Hide, crossterm::event::EnableMouseCapture)?;

    let mut center_x = (cfg.min + cfg.max) as f64 / 2.0;
    let mut center_y = (cfg.min + cfg.max) as f64 / 2.0;
    let world_w = (cfg.max - cfg.min + 1) as f64;
    let world_h = (cfg.max - cfg.min + 1) as f64;

    // base scale: cells per world unit
    let (term_w, term_h) = terminal::size()?;
    let cols = term_w as f64;
    let rows = (term_h.saturating_sub(2)) as f64;
    let base_scale = (cols / world_w).min(rows / world_h).max(0.0001);
    let mut zoom = 1.0f64; // multiplier for base_scale

    let mut dragging = false;
    let mut last_mouse: Option<(i16, i16)> = None;

    let mut should_quit = false;
    // start with a full clear
    execute!(stdout, terminal::Clear(terminal::ClearType::All), cursor::MoveTo(0,0))?;
    let mut redraw = true;
    while !should_quit {
        // wait for an event (longer poll) to reduce redraw frequency and flicker
        if event::poll(Duration::from_millis(200))? {
            match event::read()? {
                Event::Key(k) => match k.code {
                    KeyCode::Char('q') => { should_quit = true; }
                    KeyCode::Char('+') | KeyCode::Char('=') => {
                        // zoom while keeping the world point under the anchor (mouse if known,
                        // otherwise screen center) fixed so the map doesn't visually shift.
                        let (tw, th) = terminal::size()?;
                        let cols_f = tw as f64;
                        let rows_f = (th.saturating_sub(2)) as f64;
                        let char_aspect_local = if char_aspect > 0.0 {
                            char_aspect
                        } else {
                            ((cols_f / rows_f) * 0.5).max(0.8)
                        };
                        let scale_x_before = (cols_f / world_w).max(1e-6) * zoom;
                        let scale_y_before = (rows_f / world_h).max(1e-6) * zoom / char_aspect_local;
                        let anchor_x = if let Some((mx, _)) = last_mouse { mx as f64 } else { cols_f / 2.0 };
                        let anchor_y = if let Some((_, my)) = last_mouse { my as f64 } else { rows_f / 2.0 };
                        let world_anchor_x = center_x + (anchor_x - cols_f / 2.0) / scale_x_before;
                        let world_anchor_y = center_y + (anchor_y - rows_f / 2.0) / scale_y_before;
                        zoom *= 1.2;
                        let scale_x_after = (cols_f / world_w).max(1e-6) * zoom;
                        let scale_y_after = (rows_f / world_h).max(1e-6) * zoom / char_aspect_local;
                        center_x = world_anchor_x - (anchor_x - cols_f / 2.0) / scale_x_after;
                        center_y = world_anchor_y - (anchor_y - rows_f / 2.0) / scale_y_after;
                    }
                    KeyCode::Char('-') => {
                        let (tw, th) = terminal::size()?;
                        let cols_f = tw as f64;
                        let rows_f = (th.saturating_sub(2)) as f64;
                        let char_aspect_local = if char_aspect > 0.0 {
                            char_aspect
                        } else {
                            ((cols_f / rows_f) * 0.5).max(0.8)
                        };
                        let scale_x_before = (cols_f / world_w).max(1e-6) * zoom;
                        let scale_y_before = (rows_f / world_h).max(1e-6) * zoom / char_aspect_local;
                        let anchor_x = if let Some((mx, _)) = last_mouse { mx as f64 } else { cols_f / 2.0 };
                        let anchor_y = if let Some((_, my)) = last_mouse { my as f64 } else { rows_f / 2.0 };
                        let world_anchor_x = center_x + (anchor_x - cols_f / 2.0) / scale_x_before;
                        let world_anchor_y = center_y + (anchor_y - rows_f / 2.0) / scale_y_before;
                        zoom /= 1.2;
                        let scale_x_after = (cols_f / world_w).max(1e-6) * zoom;
                        let scale_y_after = (rows_f / world_h).max(1e-6) * zoom / char_aspect_local;
                        center_x = world_anchor_x - (anchor_x - cols_f / 2.0) / scale_x_after;
                        center_y = world_anchor_y - (anchor_y - rows_f / 2.0) / scale_y_after;
                    }
                    KeyCode::Left => { center_x -= 10.0 / (base_scale * zoom); }
                    KeyCode::Right => { center_x += 10.0 / (base_scale * zoom); }
                    KeyCode::Up => { center_y -= 5.0 / (base_scale * zoom); }
                    KeyCode::Down => { center_y += 5.0 / (base_scale * zoom); }
                    _ => {}
                },
                Event::Mouse(me) => {
                    match me.kind {
                        MouseEventKind::ScrollUp => {
                            // zoom toward mouse
                            let mx = me.column as f64;
                            let my = me.row as f64;
                            let scale_before = base_scale * zoom;
                            let world_mouse_x = center_x + (mx - cols/2.0) / scale_before;
                            let world_mouse_y = center_y + (my - rows/2.0) / scale_before;
                            zoom *= 1.2;
                            let scale_after = base_scale * zoom;
                            center_x = world_mouse_x - (mx - cols/2.0) / scale_after;
                            center_y = world_mouse_y - (my - rows/2.0) / scale_after;
                        }
                        MouseEventKind::ScrollDown => {
                            let mx = me.column as f64;
                            let my = me.row as f64;
                            let scale_before = base_scale * zoom;
                            let world_mouse_x = center_x + (mx - cols/2.0) / scale_before;
                            let world_mouse_y = center_y + (my - rows/2.0) / scale_before;
                            zoom /= 1.2;
                            let scale_after = base_scale * zoom;
                            center_x = world_mouse_x - (mx - cols/2.0) / scale_after;
                            center_y = world_mouse_y - (my - rows/2.0) / scale_after;
                        }
                        MouseEventKind::Down(MouseButton::Left) => {
                            dragging = true;
                            last_mouse = Some((me.column as i16, me.row as i16));
                        }
                        MouseEventKind::Up(MouseButton::Left) => {
                            dragging = false;
                            last_mouse = None;
                        }
                        MouseEventKind::Drag(MouseButton::Left) => {
                            if dragging {
                                if let Some((lx, ly)) = last_mouse {
                                    let nx = me.column as i16;
                                    let ny = me.row as i16;
                                    let dx = nx - lx;
                                    let dy = ny - ly;
                                    let scale = base_scale * zoom;
                                    center_x -= (dx as f64) / scale;
                                    center_y -= (dy as f64) / scale;
                                    last_mouse = Some((nx, ny));
                                } else {
                                    last_mouse = Some((me.column as i16, me.row as i16));
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Event::Resize(w, h) => {
                    // recompute cols/rows and base_scale
                    let cols_f = w as f64;
                    let rows_f = (h.saturating_sub(2)) as f64;
                    let _ = (cols_f, rows_f);
                    // we'll recompute scale on next draw
                }
                _ => {}
            }
            redraw = true;
        }

        if !redraw {
            continue;
        }
        redraw = false;

        // draw current view
        let (tw, th) = terminal::size()?;
        let cols_i = tw as usize;
        let rows_i = (th.saturating_sub(2)) as usize;
        let cols_f = cols_i as f64;
        let rows_f = rows_i as f64;
        // If user passed char_aspect <= 0.0 we auto-compute an aspect factor from terminal
        // and world dimensions so circles stay roughly circular as terminal size changes.
        let char_aspect_local = if char_aspect > 0.0 {
            char_aspect
        } else {
            // Auto-compute a character height/width aspect ratio.
            // Use cols/rows (wide terminals -> larger aspect) multiplied by a
            // tunable base so the resulting value is > 1.0 for typical monospace fonts.
            // This causes vertical scale to be reduced (divide by char_aspect_local)
            // so shapes remain approximately circular.
            ((cols_f / rows_f) * 0.5).max(0.8)
        };
        // account for terminal character aspect ratio: characters are typically taller than wide
        // use char_aspect_local to squash vertical scale so circles look round
        let scale_x = (cols_f / world_w).max(1e-6) * zoom;
        let scale_y = (rows_f / world_h).max(1e-6) * zoom / char_aspect_local;

        use std::io::Write;
        execute!(stdout, cursor::MoveTo(0,0))?;
        if use_braille {
            // Braille: each cell represents 2x4 dots. Build a high-res buffer then map.
            let braille_w = cols_i * 2;
            let braille_h = rows_i * 4;
            let mut high: Vec<Vec<u8>> = vec![vec![0u8; braille_w]; braille_h];

            // Compute scale in dots-per-world-unit directly to avoid double-scaling
            // and to respect the auto-computed char aspect.
            let dots_w = (cols_f * 2.0).max(1.0);
            let dots_h = (rows_f * 4.0).max(1.0);
            let dot_scale_x = (dots_w / world_w).max(1e-12) * zoom;
            let dot_scale_y = (dots_h / world_h).max(1e-12) * zoom / char_aspect_local;

            for &(x, y) in kept.iter() {
                // map into high-res dot coordinates (floating)
                let sx = (x - center_x) * dot_scale_x + dots_w / 2.0;
                let sy = (y - center_y) * dot_scale_y + dots_h / 2.0;
                let bx_f = sx.floor();
                let by_f = sy.floor();
                let bx = bx_f as isize;
                let by = by_f as isize;
                if bx < -1 || by < -1 || bx as usize >= braille_w + 1 || by as usize >= braille_h + 1 {
                    continue;
                }
                // Bilinear splat to neighboring dots to reduce banding/artifacts.
                let fx = (sx - bx_f).clamp(0.0, 1.0);
                let fy = (sy - by_f).clamp(0.0, 1.0);
                let w00 = (1.0 - fx) * (1.0 - fy);
                let w10 = fx * (1.0 - fy);
                let w01 = (1.0 - fx) * fy;
                let w11 = fx * fy;
                // scale weights so at least one neighbor receives a non-zero increment
                // use a small multiplier to keep values in 0..4 range
                let scale_w = 4.0;
                let add00 = (w00 * scale_w).round() as u8;
                let add10 = (w10 * scale_w).round() as u8;
                let add01 = (w01 * scale_w).round() as u8;
                let add11 = (w11 * scale_w).round() as u8;

                // helper to safely add to high-res buffer
                let mut safe_add = |xx: isize, yy: isize, add: u8| {
                    if add == 0 { return; }
                    if xx >= 0 && yy >= 0 {
                        let ux = xx as usize;
                        let uy = yy as usize;
                        if ux < braille_w && uy < braille_h {
                            high[uy][ux] = high[uy][ux].saturating_add(add);
                        }
                    }
                };

                safe_add(bx,     by,     add00);
                safe_add(bx + 1, by,     add10);
                safe_add(bx,     by + 1, add01);
                safe_add(bx + 1, by + 1, add11);
            }

            // convert high-res to braille chars
            let mut out = String::with_capacity((cols_i + 1) * rows_i + 128);
            for by in 0..rows_i {
                for bx in 0..cols_i {
                    // each braille cell covers 2x4 dots at high coords (hx = bx*2..bx*2+1, hy = by*4..by*4+3)
                    let mut mask = 0u8;
                    // mapping to braille dot numbers
                    for dy in 0..4 {
                        for dx in 0..2 {
                            let hx = bx * 2 + dx;
                            let hy = by * 4 + dy;
                            let v = high[hy][hx];
                            if v > 0 {
                                // determine dot number
                                let dot = match (dx, dy) {
                                    (0,0) => 1,
                                    (0,1) => 2,
                                    (0,2) => 3,
                                    (0,3) => 7,
                                    (1,0) => 4,
                                    (1,1) => 5,
                                    (1,2) => 6,
                                    (1,3) => 8,
                                    _ => 0,
                                };
                                if dot > 0 {
                                    mask |= 1 << (dot - 1);
                                }
                            }
                        }
                    }
                    if mask == 0 {
                        out.push(' ');
                    } else {
                        let ch = std::char::from_u32(0x2800u32 + (mask as u32)).unwrap_or(' ');
                        out.push(ch);
                    }
                }
                out.push('\n');
            }
            out.push_str(&format!("Stars: {}  center: ({:.2},{:.2})  zoom: {:.2}  (q to quit)\n", kept.len(), center_x, center_y, zoom));
            stdout.write_all(out.as_bytes())?;
            stdout.flush()?;
        } else {
            let mut buf: Vec<Vec<u32>> = vec![vec![0u32; cols_i]; rows_i];
            for &(x, y) in kept.iter() {
                let sx = (x - center_x) * scale_x + cols_f / 2.0;
                let sy = (y - center_y) * scale_y + rows_f / 2.0;
                let cx = sx.floor() as isize;
                let cy = sy.floor() as isize;
                if cx >= 0 && (cx as usize) < cols_i && cy >= 0 && (cy as usize) < rows_i {
                    buf[cy as usize][cx as usize] = buf[cy as usize][cx as usize].saturating_add(1);
                }
            }
            let mut out = String::with_capacity((cols_i + 1) * rows_i + 80);
            for row in 0..rows_i {
                for col in 0..cols_i {
                    let v = buf[row][col];
                    let ch = if v == 0 { ' ' } else if v < 3 { '.' } else if v < 8 { '*' } else { '@' };
                    out.push(ch);
                }
                out.push('\n');
            }
            out.push_str(&format!("Stars: {}  center: ({:.2},{:.2})  zoom: {:.2}  (q to quit)\n", kept.len(), center_x, center_y, zoom));
            stdout.write_all(out.as_bytes())?;
            stdout.flush()?;
        }
    }

    // restore terminal
    execute!(stdout, crossterm::event::DisableMouseCapture, cursor::Show, terminal::LeaveAlternateScreen)?;
    terminal::disable_raw_mode()?;
    Ok(())
}
