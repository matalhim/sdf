import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from multiprocessing import Pool
from corsikaio import CorsikaFile
from config import prim_particle, theta
from functions import (
    parse_metadata,
    parse_corsika_showers,
    save_event_data,
)

input_folder = prim_particle + f'{theta}'
input_path = os.path.join(os.path.dirname(__file__), 'data', input_folder)
output_path = os.path.join(os.path.dirname(__file__), 'output', input_folder)
os.makedirs(output_path, exist_ok=True)

def process_file(filename):
    if not (filename.startswith('pDAT') and not filename.endswith('.dbase')):
        return

    file_path = os.path.join(input_path, filename)
    metadata_filename = f'{filename}.dbase'
    metadata_file_path = os.path.join(input_path, metadata_filename)
    save_path = os.path.join(output_path, f'{filename}')
    os.makedirs(save_path, exist_ok=True)

    with CorsikaFile(file_path) as f:
        events = list(f)
        metadata = parse_metadata(metadata_file_path)

        for event_idx, event in tqdm(
            enumerate(events),
            total=len(events),
            desc=f"Файл {filename}",
            position=1,
            leave=False,
            colour="green"
        ):
            header = event.header
            particles = event.data

            if isinstance(header, np.void):
                header = {field: header[field] for field in header.dtype.names}

            header.update(metadata)
            save_event_data(event_idx, particles, header, save_path)

    params = parse_corsika_showers(file_path)
    df = pd.DataFrame.from_dict(params, orient='index')
    df.index.name = 'event_id'
    df.reset_index(inplace=True)
    outfile = os.path.join(save_path, '_params.csv')
    df.to_csv(outfile, index=False, encoding='utf-8')

if __name__ == "__main__":
    all_files = os.listdir(input_path)
    all_files = [f for f in all_files if f.startswith('pDAT') and not f.endswith('.dbase')]

    process_map(process_file, all_files, max_workers=os.cpu_count(), desc="Обработка файлов")
