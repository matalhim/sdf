from tqdm import tqdm
import os
import numpy as np
import pandas as pd
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
    current_event = []
    index = 0

    while index < len(data):
        if data[index] == -1:
            if current_event:
                events.append(current_event)
                current_event = []
            index += 1
            continue

        scalars = list(data[index:index + 35])
        index += 35

        arrays = {}
        for key, shape in sizes.items(): 
            size = np.prod(shape)
            arrays[key] = data[index:index + size].reshape(shape).tolist()
            index += size

        marker = data[index] if index < len(data) else -1
        index += 1

        event = scalars + [arrays[key] for key in sizes] + [marker]
        events.append(event)

    df = pd.DataFrame(events, columns=columns)  
    output_path = os.path.join(output_base_path, f'{file_name[:5]}.csv')
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Processed and saved: {output_path}")
