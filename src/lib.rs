use pyo3::prelude::*;
use std::collections::HashMap;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

#[pyfunction]
fn fast_process_signals(
    highs: Vec<f64>, 
    lows: Vec<f64>, 
    lookback: usize
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let n = highs.len();
    let mut slopes_high = vec![0.0; n];
    let mut slopes_low = vec![0.0; n];

    // Recorremos los datos para calcular las regresiones (Banderas)
    for i in lookback..n {
        let x_sum: f64 = (0..lookback).map(|x| x as f64).sum();
        let xx_sum: f64 = (0..lookback).map(|x| (x * x) as f64).sum();
        let n_f = lookback as f64;
        let denominator = n_f * xx_sum - x_sum * x_sum;

        if denominator != 0.0 {
            // Regresión para Highs
            let xy_sum_h: f64 = (0..lookback).map(|j| (j as f64) * highs[i - lookback + j]).sum();
            let y_sum_h: f64 = (i - lookback..i).map(|j| highs[j]).sum();
            slopes_high[i] = (n_f * xy_sum_h - x_sum * y_sum_h) / denominator;

            // Regresión para Lows
            let xy_sum_l: f64 = (0..lookback).map(|j| (j as f64) * lows[i - lookback + j]).sum();
            let y_sum_l: f64 = (i - lookback..i).map(|j| lows[j]).sum();
            slopes_low[i] = (n_f * xy_sum_l - x_sum * y_sum_l) / denominator;
        }
    }

    Ok((slopes_high, slopes_low))
}

// herrramientas para el calculo de soportes y resistencia
#[pyfunction]
fn get_key_levels(highs: Vec<f64>, lows: Vec<f64>, bin_size: f64) -> PyResult<Vec<f64>> {
    let mut counts: HashMap<i64, i32> = HashMap::new();

    // 1. Detectar Pivots y agrupar en rangos (bins)
    // Usamos una ventana simple de 3 velas para identificar picos
    for i in 2..(highs.len() - 2) {
        // Pivot High
        if highs[i] > highs[i-1] && highs[i] > highs[i-2] && highs[i] > highs[i+1] && highs[i] > highs[i+2] {
            let bin = (highs[i] / bin_size).round() as i64;
            *counts.entry(bin).or_insert(0) += 1;
        }
        // Pivot Low
        if lows[i] < lows[i-1] && lows[i] < lows[i-2] && lows[i] < lows[i+1] && lows[i] < lows[i+2] {
            let bin = (lows[i] / bin_size).round() as i64;
            *counts.entry(bin).or_insert(0) += 1;
        }
    }

    // 2. Filtrar solo los niveles con más de 2 "toques" (frecuencia)
    let mut levels: Vec<f64> = counts.into_iter()
        .filter(|&(_, count)| count >= 2)
        .map(|(bin, _)| (bin as f64) * bin_size)
        .collect();

    levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(levels)
}

// herramientas para los modelos de machine learning 

#[pyfunction]
fn create_sequences(data: Vec<f64>, time_step: usize) -> PyResult<(Vec<Vec<f64>>, Vec<f64>)> {
    let mut x_train = Vec::with_capacity(data.len() - time_step);
    let mut y_train = Vec::with_capacity(data.len() - time_step);

    if data.len() <= time_step {
        return Ok((x_train, y_train));
    }

    // Ventaneo ultrarrápido en Rust
    for i in time_step..data.len() {
        let window = data[i - time_step..i].to_vec();
        x_train.push(window);
        y_train.push(data[i]);
    }

    Ok((x_train, y_train))
}


// herramientas para el calculo de breakout indicator
#[pyfunction]
fn detect_channel_breakout(highs: Vec<f64>, lows: Vec<f64>, closes: Vec<f64>, backcandles: usize) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let n = highs.len();
    let mut slope_lows = vec![0.0; n];
    let mut interc_lows = vec![0.0; n];
    let mut slope_highs = vec![0.0; n];
    let mut interc_highs = vec![0.0; n];

    for i in backcandles..n {
        let start = i - backcandles;
        let x: Vec<f64> = (0..backcandles).map(|val| val as f64).collect();
        
        // Regresión para mínimos (Soporte)
        let y_lows = &lows[start..i];
        let (s_l, i_l) = simple_lin_reg(&x, y_lows);
        
        // Regresión para máximos (Resistencia)
        let y_highs = &highs[start..i];
        let (s_h, i_h) = simple_lin_reg(&x, y_highs);

        slope_lows[i] = s_l;
        interc_lows[i] = i_l;
        slope_highs[i] = s_h;
        interc_highs[i] = i_h;
    }

    Ok((slope_lows, interc_lows, slope_highs, interc_highs))
}

// Función auxiliar de regresión lineal simple
fn simple_lin_reg(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let sum_x = x.iter().sum::<f64>();
    let sum_y = y.iter().sum::<f64>();
    let sum_xy = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum::<f64>();
    let sum_xx = x.iter().map(|xi| xi * xi).sum::<f64>();

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;
    (slope, intercept)
}


// herramienta de modelo Montecarlo 
#[pyfunction]
fn monte_carlo_simulation(
    initial_price: f64,
    mu: f64,    // Retorno promedio diario
    sigma: f64, // Volatilidad diaria
    days: usize,
    num_simulations: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut all_paths = Vec::with_capacity(num_simulations);

    for _ in 0..num_simulations {
        let mut path = Vec::with_capacity(days + 1);
        path.push(initial_price);
        let mut current_price = initial_price;

        for _ in 0..days {
            let z = normal.sample(&mut rng);
            // Fórmula: St = St-1 * exp((mu - 0.5 * sigma^2) + sigma * Z)
            let drift = mu - 0.5 * sigma.powi(2);
            let shock = sigma * z;
            current_price *= (drift + shock).exp();
            path.push(current_price);
        }
        all_paths.push(path);
    }

    Ok(all_paths)
}

// herramientas para el modelo de correlación de activos 
#[pyfunction]
fn rolling_correlation(x: Vec<f64>, y: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    let n = x.len();
    let mut results = vec![0.0; n];

    for i in window..n {
        let slice_x = &x[i - window..i];
        let slice_y = &y[i - window..i];
        
        let mean_x = slice_x.iter().sum::<f64>() / window as f64;
        let mean_y = slice_y.iter().sum::<f64>() / window as f64;

        let mut num = 0.0;
        let mut den_x = 0.0;
        let mut den_y = 0.0;

        for j in 0..window {
            let diff_x = slice_x[j] - mean_x;
            let diff_y = slice_y[j] - mean_y;
            num += diff_x * diff_y;
            den_x += diff_x.powi(2);
            den_y += diff_y.powi(2);
        }

        let r = num / (den_x.sqrt() * den_y.sqrt());
        results[i] = if r.is_nan() { 0.0 } else { r };
    }

    Ok(results)
}


#[pymodule]
fn signal_market_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_process_signals, m)?)?;
    m.add_function(wrap_pyfunction!(get_key_levels, m)?)?;
    m.add_function(wrap_pyfunction!(create_sequences, m)?)?; 
    m.add_function(wrap_pyfunction!(detect_channel_breakout, m)?)?;
    m.add_function(wrap_pyfunction!(monte_carlo_simulation, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_correlation, m)?)?;
    Ok(())
}