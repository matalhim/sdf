from __future__ import annotations
from tqdm import tqdm
import os
import numpy as np
import polars as pl
from config import (
    prim_particle,
    theta,
    sizes,
    columns,
)

input_folder = prim_particle + f'{theta}'
input_path = os.path.join(os.path.dirname(__file__), 'data', input_folder)
output_base_path = os.path.join(os.path.dirname(__file__), 'output', input_folder)

os.makedirs(output_base_path, exist_ok=True)

files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]

for file_name in tqdm(files, desc="Processing files", unit="file"):
    file_path = os.path.join(input_path, file_name)
    data = np.fromfile(file_path, dtype=np.float32)

    events = []
    index = 0

    while index < len(data):
        if data[index] == -1:
            index += 1
            continue

        scalars = list(data[index:index + 35])
        index += 35

        arrays = []
        for key, shape in sizes.items():
            size = np.prod(shape)
            arr = data[index:index + size].reshape(shape).tolist()
            arrays.append(arr)
            index += size

        marker = data[index] if index < len(data) else -1
        index += 1

        event = scalars + arrays + [marker]
        events.append(event)

    df = pl.DataFrame(schema_overrides={col: pl.Object for col in columns})
    df = pl.DataFrame(events, schema=columns, orient="row")

    output_path = os.path.join(output_base_path, f'{file_name[:10]}.parquet')
    df.write_parquet(output_path)
    print(f"Processed and saved: {output_path}")
