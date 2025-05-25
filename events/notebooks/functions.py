from __future__ import annotations
import numpy as np
import ast
from scipy.stats import mode
from numba import njit
from modeling_parameters.config import (
    CENTRAL_STATIONS,
)


def ast_df(df):
    df['station_flags'] = df['station_flags'].apply(ast.literal_eval)
    df['worked_clusters'] = df['worked_clusters'].apply(ast.literal_eval)
    df['rho'] = df['rho'].apply(ast.literal_eval)
    
def ast_modeling_df(df):
    df['r'] = df['r'].apply(ast.literal_eval)
    df['worked_stations'] = df['worked_stations'].apply(ast.literal_eval)
    df['worked_clusters'] = df['worked_clusters'].apply(ast.literal_eval)
    
    df['rho'] = df['rho'].apply(ast.literal_eval)
    
    
def is_central_stations(row, n):
    station_flags = row['station_flags']
    rho = row['rho']
    
    worked_station_indices = [i for i, flag in enumerate(station_flags) if flag]

    rho_with_station_idx = list(zip(worked_station_indices, rho))
    sorted_rho = sorted(rho_with_station_idx, key=lambda x: x[1], reverse=True)

    top_n_stations = [idx for idx, _ in sorted_rho[:n]]
  
    central_count = sum(1 for idx in top_n_stations if idx in CENTRAL_STATIONS)
    
    return central_count >= n



def binned_log_stats(r_values, rho_values,  num_bins=20, r_min_val=1, r_max_val=1000):
    r_values = np.array(r_values)
    rho_values = np.array(rho_values)
    mask = r_values > 0
    r_values = r_values[mask]
    rho_values = rho_values[mask]

    r_log_bins = np.logspace(np.log10(r_min_val), np.log10(r_max_val), num_bins)
    r_bin_centers = np.sqrt(r_log_bins[:-1] * r_log_bins[1:])

    sum_rho_in_bins = np.zeros(len(r_bin_centers))
    count_in_bins = np.zeros(len(r_bin_centers))
    sum_rho_sq_in_bins = np.zeros(len(r_bin_centers))
    median_rho_in_bins = np.full(len(r_bin_centers), np.nan)

    rho_values_per_bin = [[] for _ in range(len(r_bin_centers))]

    # Заполняем бины
    for r_val, rho_val in zip(r_values, rho_values):
        bin_idx = np.searchsorted(r_log_bins, r_val) - 1
        if 0 <= bin_idx < len(r_bin_centers):
            sum_rho_in_bins[bin_idx] += rho_val
            sum_rho_sq_in_bins[bin_idx] += rho_val**2
            count_in_bins[bin_idx] += 1
            rho_values_per_bin[bin_idx].append(rho_val)

    valid_bins = count_in_bins > 0

    mean_rho_in_bins = np.zeros_like(sum_rho_in_bins)
    std_rho_in_bins = np.zeros_like(sum_rho_sq_in_bins)

    mean_rho_in_bins[valid_bins] = sum_rho_in_bins[valid_bins] / count_in_bins[valid_bins]
    std_rho_in_bins[valid_bins] = np.sqrt(
        sum_rho_sq_in_bins[valid_bins] / count_in_bins[valid_bins] - mean_rho_in_bins[valid_bins]**2
    )
    

    for i in range(len(r_bin_centers)):
        if count_in_bins[i] > 0:
            median_rho_in_bins[i] = np.median(rho_values_per_bin[i])

    log_mean_rho = np.log10(mean_rho_in_bins[valid_bins])
    log_std_rho = std_rho_in_bins[valid_bins] / (mean_rho_in_bins[valid_bins] * np.log(10)) 
    # log_std_rho = std_rho_in_bins[valid_bins] / (np.sqrt(count_in_bins[valid_bins]) * mean_rho_in_bins[valid_bins] * np.log(10))

    
    yerr_lower_vals = mean_rho_in_bins[valid_bins] - 10**(log_mean_rho - log_std_rho)
    yerr_upper_vals = 10**(log_mean_rho + log_std_rho) - mean_rho_in_bins[valid_bins]

    yerr_lower_vals = np.abs(yerr_lower_vals)
    yerr_upper_vals = np.abs(yerr_upper_vals)

    return (
        r_bin_centers[valid_bins],
        mean_rho_in_bins[valid_bins],
        [yerr_lower_vals, yerr_upper_vals],
        median_rho_in_bins[valid_bins],
        
    )
