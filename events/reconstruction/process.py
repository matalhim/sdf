import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from modeling_parameters.reconstruction.geant.functions import optimize_parameters, filter_clusters
from modeling_parameters.config import COORDINATES_PATH

REAL_EVENTS_CSV = r"D:/github/repositories/sdf/events/output/events.csv"
OUTPUT_CSV = r"D:/github/repositories/sdf/events/output/events_reconstruction.csv"
RHO_DIVISOR = 14.7 * 4 * 2.56


def parse_json_list(val):
    try:
        return json.loads(val)
    except:
        return []


def load_coordinates(path):
    df = pd.read_csv(path)
    return df['X'].values, df['Y'].values, df['Z'].values, df


def process_single_event(args):
    idx, row, X, Y, Z, coords_df = args
    theta = row['Theta']
    phi = row['Phi']
    q_std = row['q_std']  


    station_flags = np.array([q is not None for q in q_std], dtype=bool)

    worked_clusters = []
    for cl in range(1, 10):
        start, end = (cl - 1) * 4, cl * 4
        if station_flags[start:end].sum() >= 2:
            worked_clusters.append(cl)
    if not worked_clusters:
        return None

    rho_true = np.array([
        (q / RHO_DIVISOR) if (flag and q is not None) else 0.0
        for q, flag in zip(q_std, station_flags)
    ], dtype=float)

    coords_masked = coords_df.iloc[station_flags]
    X_det = coords_masked['X'].values
    Y_det = coords_masked['Y'].values
    rho_det = rho_true[station_flags]


    Z0 = 0 
    X0_opt, Y0_opt, Ne_opt, s_opt, loss = optimize_parameters(
        X_det, Y_det, rho_det, theta, phi, Z0, return_loss=True
    )

    return {
        'event_index': idx,
        'worked_clusters': worked_clusters,
        'station_flags': station_flags.tolist(),
        'rho': rho_det.tolist(),
        'X0_opt': X0_opt,
        'Y0_opt': Y0_opt,
        'Ne_opt': Ne_opt,
        's_opt': s_opt,
        'loss': loss,
    }

# Главная функция

def main():
    X, Y, Z, coords_df = load_coordinates(COORDINATES_PATH)

    df = pd.read_csv(
        REAL_EVENTS_CSV,
        converters={
            'q_std': parse_json_list,
            'Theta': float,
            'Phi': float,
        }
    )

    args = [
        (i, row, X, Y, Z, coords_df)
        for i, row in df.iterrows()
    ]

    results = []
    with tqdm(total=len(args), desc="Processing real events") as pbar:
        with ProcessPoolExecutor() as executor:
            for res in executor.map(process_single_event, args):
                if res is not None:
                    results.append(res)
                pbar.update(1)

    result_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
