import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.special import gamma, gammaln
from tqdm import tqdm
from numba import njit


def compute_r(X0, Y0, Z0, theta, phi, X_det, Y_det, Z_det):
    """Векторизованная версия compute_r без Numba"""
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)
    
    vx = np.cos(phi_rad) * np.sin(theta_rad)
    vy = np.sin(phi_rad) * np.sin(theta_rad)
    vz = np.cos(theta_rad)
    
    t = ((X_det - X0) * vx + (Y_det - Y0) * vy + (Z_det - Z0) * vz)
    P_ix = X0 + t * vx
    P_iy = Y0 + t * vy
    P_iz = Z0 + t * vz
    
    return np.sqrt((X_det - P_ix)**2 + (Y_det - P_iy)**2 + (Z_det - P_iz)**2)


def rho_model(r, Ne, s, r_m=79):
    term1 = Ne / (2 * np.pi * r_m**2)
    term2 = (r / r_m) ** (s - 2)
    term3 = (1 + r / r_m) ** (s - 4.5)
    term4 = gamma(4.5 - s) / (gamma(s) * gamma(4.5 - 2 * s))
    return term1 * term2 * term3 * term4


def log_rho_model(r, Ne, s, r_m=80):
    log_term1 = np.log(Ne) - np.log(2 * np.pi * r_m**2)
    log_term2 = (s - 2) * np.log(r / r_m)
    log_term3 = (s - 4.5) * np.log(1 + r / r_m)
    log_term4 = gammaln(4.5 - s) - gammaln(s) - gammaln(4.5 - 2 * s)
    return log_term1 + log_term2 + log_term3 + log_term4


def rho_model_exp(r, Ne, s, r_m=80):
    return np.exp(log_rho_model(r, Ne, s, r_m))


def loss_function(params, coordinates_df, rho, theta, phi, Z0):
    X0, Y0, Ne, s = params
    r = compute_r(X0, Y0, Z0, theta, phi, coordinates_df)
    rho_calc = rho_model(r, Ne, s)
    # return np.mean((rho_calc - rho) ** 2)
    return np.mean((np.log(rho_calc) - np.log(rho)) ** 2)


def loss_function_s(params, X0, Y0, Ne, coordinates_df, rho, theta, phi, Z0):
    s = params
    r = compute_r(X0, Y0, Z0, theta, phi, coordinates_df)
    rho_calc = rho_model_exp(r, Ne, s)
    return np.mean((np.log(rho_calc) - np.log(rho)) ** 2)


def loss_function_Ne(params, X0, Y0, s, coordinates_df, rho, theta, phi, Z0):
    Ne = params
    r = compute_r(X0, Y0, Z0, theta, phi, coordinates_df)
    rho_calc = rho_model(r, Ne, s)
    return np.mean((np.log(rho_calc) - np.log(rho)) ** 2)


@njit(fastmath=True)
def compute_r_fast(X0, Y0, Z0, theta, phi, X_det, Y_det, Z_det, c=2.99e8):
    """
    Ускоренная версия compute_r с Numba.
    
    Параметры:
        X0, Y0, Z0: координаты оси ШАЛ (м)
        theta, phi: углы (град)
        X_det, Y_det, Z_det: массивы координат детекторов (м)
        c: скорость света (м/с)
    
    Возвращает:
        Массив расстояний r (м).
    """
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)
    
    # Направляющий вектор оси ШАЛ
    vx = np.cos(phi_rad) * np.sin(theta_rad)
    vy = np.sin(phi_rad) * np.sin(theta_rad)
    vz = np.cos(theta_rad)
    
    # Координаты точки Q (ось ШАЛ)
    Qx, Qy, Qz = X0, Y0, Z0
    
    r = np.zeros(len(X_det))
    
    for i in range(len(X_det)):
        Px, Py, Pz = X_det[i], Y_det[i], Z_det[i]
        
        # Вычисляем t (параметр пересечения)
        t = ((Px - Qx) * vx + (Py - Qy) * vy + (Pz - Qz) * vz) / c
        
        # Точка пересечения плоскости ШАЛ с детектором
        P_ix = Qx + t * c * vx
        P_iy = Qy + t * c * vy
        P_iz = Qz + t * c * vz
        
        # Расстояние между детектором и точкой пересечения
        dx = Px - P_ix
        dy = Py - P_iy
        dz = Pz - P_iz
        r[i] = np.sqrt(dx**2 + dy**2 + dz**2)
    
    return r

def add_noise(rho, noise_level=0.1, threshold=None):
    """Добавляет гауссов шум к значениям rho"""
    noise = np.random.normal(loc=0, scale=noise_level *
                             np.abs(rho), size=rho.shape)
    rho_noisy = rho + noise
    if threshold is not None:
        rho_noisy = np.minimum(rho_noisy, threshold)
        
    return rho_noisy

def compute_distance_from_axis(X0, Y0, theta, phi, X_det, Y_det):
    """
    Вычисляет расстояние от оси (X0,Y0) до станции (X_det,Y_det)
    при условии Z0 = Z_det (все Z координаты равны)
    
    Параметры:
        X0, Y0 - координаты оси
        theta, phi - углы направления оси (в градусах)
        X_det, Y_det - координаты станций
        
    Возвращает:
        Массив расстояний от оси до каждой станции
    """
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)
    
    vx = np.cos(phi_rad) * np.sin(theta_rad)
    vy = np.sin(phi_rad) * np.sin(theta_rad)
    
    t = ((X_det - X0) * vx + (Y_det - Y0) * vy)
    
    P_ix = X0 + t * vx
    P_iy = Y0 + t * vy
    
    return np.sqrt((X_det - P_ix)**2 + (Y_det - P_iy)**2)
