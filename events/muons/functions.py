from __future__ import annotations
import numpy as np
import pandas as pd
from modeling_parameters.config import (
    CENTRAL_STATIONS,
)


def add_distance_to_df(df, dot,
                         theta_col='Theta',
                         phi_col='Phi',
                         x0_col='X0_opt',
                         y0_col='Y0_opt',
                         output_col='r_decor'):
    """
   Считаем расстояние  от заданной точки (X_d, Y_d, Z_d)
    до прямой, проходящей через (X0_opt, Y0_opt, 0) с направлением Theta, Phi (в градусах).

    - df: pandas.DataFrame
    - X_d, Y_d, Z_d: заданная точка
    - theta_col, phi_col: названия столбцов с углами (в градусах)
    - x0_col, y0_col: координаты точки на прямой (Z=0)
    """
    X_d, Y_d, Z_d = dot
    theta_rad = np.radians(df[theta_col])
    phi_rad = np.radians(df[phi_col])

    vx = np.sin(theta_rad) * np.cos(phi_rad)
    vy = np.sin(theta_rad) * np.sin(phi_rad)
    vz = np.cos(theta_rad)
    
    dx = X_d - df[x0_col]
    dy = Y_d - df[y0_col]
    dz = Z_d - 0.0  
    
    cross_x = dy * vz - dz * vy
    cross_y = dz * vx - dx * vz
    cross_z = dx * vy - dy * vx
    
    numerator = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
    denominator = np.sqrt(vx**2 + vy**2 + vz**2)
    
    df[output_col] = numerator / denominator
    
    return df

def is_central_stations(row, n):
    station_flags = row['station_flags']
    rho = row['rho']
    
    worked_station_indices = [i for i, flag in enumerate(station_flags) if flag]

    rho_with_station_idx = list(zip(worked_station_indices, rho))
    sorted_rho = sorted(rho_with_station_idx, key=lambda x: x[1], reverse=True)

    top_n_stations = [idx for idx, _ in sorted_rho[:n]]
  
    central_count = sum(1 for idx in top_n_stations if idx in CENTRAL_STATIONS)
    
    return central_count >= n
