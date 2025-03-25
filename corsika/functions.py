import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.special import gamma, gammaln
from tqdm import tqdm


def compute_r(X0, Y0, Z0, theta, phi, coordinates_df, c=2.99e8):
    """
    Вычисляет перпендикулярное расстояние от детектора до оси ШАЛ в момент пересечения
    плоскости ШАЛ с детектором.

    Параметры:
      X0, Y0, Z0: координаты оси ШАЛ на земле (в метрах)
      theta, phi: углы направления оси ШАЛ (в градусах)
      coordinates_df: DataFrame с координатами детекторов (столбцы 'X', 'Y', 'Z', в метрах)
      c: скорость света (по умолчанию 2.99e8 м/с)

    Возвращает:
      Массив расстояний r для каждого детектора.
    """
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)
    v = np.array([np.cos(phi_rad) * np.sin(theta_rad),
                  np.sin(phi_rad) * np.sin(theta_rad),
                  np.cos(theta_rad)])

    Q = np.array([X0, Y0, Z0])

    P = np.vstack(
        (coordinates_df['X'], coordinates_df['Y'], coordinates_df['Z'])).T

    t_values = np.dot(P - Q, v) / c
    P_intersect = Q + np.outer(t_values * c, v)
    r = np.linalg.norm(P - P_intersect, axis=1)

    return r


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
