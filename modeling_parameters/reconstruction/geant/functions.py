import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.special import gamma, gammaln
from modeling_parameters.reconstruction.geant.config import bounds
from numba import njit

@njit
def compute_r(X0, Y0, Z0, theta, phi, X_det, Y_det, Z_det):
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)
    
    vx = np.cos(phi_rad) * np.sin(theta_rad)
    vy = np.sin(phi_rad) * np.sin(theta_rad)
    vz = np.cos(theta_rad)
    
    t = ((X_det - X0) * vx + (Y_det - Y0) * vy + (Z_det - Z0) * vz)
    P_ix = X0 + t * vx
    P_iy = Y0 + t * vy
    P_iz = Z0 + t * vz
    
    r = np.sqrt((X_det - P_ix)**2 + (Y_det - P_iy)**2 + (Z_det - P_iz)**2)
    return r

def rho_model(r, Ne, s, r_m=79):
    term1 = Ne / (2 * np.pi * r_m**2)
    term2 = (r / r_m) ** (s - 2)
    term3 = (1 + r / r_m) ** (s - 4.5)
    term4 = gamma(4.5 - s) / (gamma(s) * gamma(4.5 - 2 * s))
    return term1 * term2 * term3 * term4

def loss_function(params, X_det, Y_det, theta, phi, Z0, rho_true):
    eps = 1e-10
    X0, Y0, Ne, s = params
    r = compute_distance_from_axis(X0, Y0, theta, phi, X_det, Y_det)
    rho_pred = rho_model(r, Ne, s)
    return np.mean((np.log(rho_pred + eps) - np.log(rho_true + eps)) ** 2)

def compute_distance_from_axis(X0, Y0, theta, phi, X_det, Y_det):
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)
    
    vx = np.cos(phi_rad) * np.sin(theta_rad)
    vy = np.sin(phi_rad) * np.sin(theta_rad)
    
    t = (X_det - X0) * vx + (Y_det - Y0) * vy
    P_ix = X0 + t * vx
    P_iy = Y0 + t * vy
    return np.sqrt((X_det - P_ix)**2 + (Y_det - P_iy)**2)

def optimize_parameters(X_det, Y_det, rho_geant, theta_g, phi_g, Z0_g, return_loss=False):
    args = (X_det, Y_det, theta_g, phi_g, Z0_g, rho_geant)
    
    result_de = differential_evolution(
        loss_function,
        bounds,
        args=args,
        strategy='best1bin',
        popsize=25,
        mutation=(0.5, 1),
        recombination=0.7,
        polish=True,
        updating='deferred',
        seed=42,
        workers=1,
    )

    result_min = minimize(
        loss_function,
        result_de.x,
        args=args,
        method='Nelder-Mead',
        bounds=bounds,
        options={'maxiter': 500}
    )

    if return_loss:
        return (*result_min.x, result_min.fun)
    else:
        return result_min.x

def filter_clusters(E_stations):
    N_e = E_stations['E'] / 8.2
    S = 0.8 * 0.8 * 4
    mask_energy = N_e > 0.75

    E_stations = E_stations.copy()
    E_stations['mask_energy'] = mask_energy
    E_stations['N_e'] = N_e

    worked_clusters = []
    worked_stations = []

    for cluster_id in sorted(E_stations['cluster'].unique()):
        cluster_stations = E_stations[E_stations['cluster'] == cluster_id]
        active_stations = cluster_stations[cluster_stations['mask_energy']]

        if len(active_stations) >= 2:
            worked_clusters.append(cluster_id)
            worked_stations.extend(active_stations.index.tolist())

    mask_final = np.zeros(len(E_stations), dtype=bool)
    mask_final[worked_stations] = True

    rho_final = (E_stations.loc[mask_final, 'N_e'] / S).values

    return worked_clusters, worked_stations, mask_final, rho_final
