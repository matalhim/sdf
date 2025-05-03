import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from modeling_parameters.config import (
    PRIM_PARTICLE, THETA, COORDINATES_PATH,
    READ_GEANT_OUTPUT_DIR, RECONSTRUCTION_GEANT_OUTPUT_DIR,
)
from modeling_parameters.reconstruction.geant.config import converters
from modeling_parameters.reconstruction.geant.functions import (
    compute_r,
    optimize_parameters,
    filter_clusters,
)

def load_coordinates(path):
    coords = pd.read_csv(path)
    return coords['X'].values, coords['Y'].values, coords['Z'].values, coords

def get_input_files(path):
    files = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.csv'):
                files.append(os.path.join(root, filename))
    return files

def prepare_event(E_1_event):
    e_list = []
    for i_cluster, cluster in enumerate(E_1_event):
        for j_station, value in enumerate(cluster):
            e_list.append({'cluster': i_cluster+1, 'station': j_station+1, 'E': value})
    return pd.DataFrame(e_list)

def process_single_event(args):
    file_df, j_event, X, Y, Z, coordinates_df = args

    E_1_event = file_df['EdepStNE'].iloc[j_event]
    theta_g = file_df['Teta'].iloc[j_event]
    phi_g = file_df['Fi'].iloc[j_event]
    X0_g = file_df['XAxisShift'].iloc[j_event]
    Y0_g = file_df['YAxisShift'].iloc[j_event]
    Z0_g = -18
    Ne_g = file_df['NeNKGlong'].iloc[j_event]
    s_g = file_df['sNKGlong'].iloc[j_event]

    r_geant = compute_r(X0_g, Y0_g, Z0_g, theta_g, phi_g, X, Y, Z)
    E_stations = prepare_event(E_1_event)

    worked_clusters, worked_stations, mask, rho_geant = filter_clusters(E_stations)

    if len(worked_clusters) == 0:
        return None

    coordinates_masked = coordinates_df.iloc[mask]
    X_det, Y_det = coordinates_masked['X'].values, coordinates_masked['Y'].values

    result = optimize_parameters(
        X_det, Y_det, rho_geant, theta_g, phi_g, Z0_g, return_loss=True
    )

    X0_g_opt, Y0_g_opt, Ne_g_opt, s_g_opt, loss_value = result

    return {
        'theta': theta_g,
        'phi': phi_g,
        'X0': X0_g,
        'Y0': Y0_g,
        'Ne': Ne_g,
        's': s_g,
        'r': r_geant.tolist(),
        'rho': rho_geant.tolist(),
        'mask': mask.tolist(),
        'worked_clusters': worked_clusters,
        'worked_stations': worked_stations,
        'X0_opt': X0_g_opt,
        'Y0_opt': Y0_g_opt,
        'Ne_opt': Ne_g_opt,
        's_opt': s_g_opt,
        'loss': loss_value,
    }

def process_file(file_name, input_path, output_path, X, Y, Z, coordinates_df, file_idx, total_files):
    file_path = os.path.join(input_path, file_name)
    file_df = pd.read_csv(file_path, converters=converters)
    # file_df = pd.read_csv(file_path, converters=converters).tail(10).reset_index(drop=True)


    os.makedirs(output_path, exist_ok=True)

    args_list = [(file_df, j_event, X, Y, Z, coordinates_df) for j_event in range(len(file_df))]

    with tqdm(total=len(args_list), desc=f"Файл {file_idx}/{total_files}: {file_name}", position=1, leave=False) as pbar:
        with ProcessPoolExecutor() as executor:
            reconstruction_results = []
            for result in executor.map(process_single_event, args_list):
                reconstruction_results.append(result)
                pbar.update(1)

    reconstruction_df = pd.DataFrame([res for res in reconstruction_results if res is not None])

    output_file_name = f'{os.path.splitext(file_name)[0]}_reconstruction.csv'
    reconstruction_df.to_csv(os.path.join(output_path, output_file_name), index=False)
    print(f"Сохранено: {output_path}/{output_file_name}")

def main():
    X, Y, Z, coordinates_df = load_coordinates(COORDINATES_PATH)
    files = get_input_files(READ_GEANT_OUTPUT_DIR)

    with tqdm(total=len(files), desc=f"{PRIM_PARTICLE}{THETA}", position=0, unit="file") as pbar:
        for idx, file_name in enumerate(files, start=1):
            process_file(file_name, READ_GEANT_OUTPUT_DIR, RECONSTRUCTION_GEANT_OUTPUT_DIR, X, Y, Z, coordinates_df, idx, len(files))
            pbar.update(1)

if __name__ == "__main__":
    main()
