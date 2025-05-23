import numpy as np
from __future__ import annotations
import ast
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
    
    # Получаем индексы сработавших станций
    worked_station_indices = [i for i, flag in enumerate(station_flags) if flag]
    
    # Связываем индексы сработавших станций с их rho
    rho_with_station_idx = list(zip(worked_station_indices, rho))
    
    # Сортируем по rho по убыванию
    sorted_rho = sorted(rho_with_station_idx, key=lambda x: x[1], reverse=True)
    
    # Берем top-n по rho
    top_n_stations = [idx for idx, _ in sorted_rho[:n]]
    
    # Считаем, сколько из них — центральные станции
    central_count = sum(1 for idx in top_n_stations if idx in CENTRAL_STATIONS)
    
    return central_count >= n


