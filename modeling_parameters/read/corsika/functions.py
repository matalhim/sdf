import pandas as pd
import numpy as np
from corsikaio import CorsikaFile
import re
import struct
import os
from config import PID_MAP


def parse_metadata(file_path):
    metadata = {}

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip() 

            match = re.search(r'#howmanyshowers#\s*(\d+)', line)
            if match:
                metadata['howmanyshowers'] = int(match.group(1))

            match = re.search(r'#energy_prim#\s*([\d\.eE\+\-]+)', line)
            if match:
                metadata['energy_prim'] = float(match.group(1))

            match = re.search(r'#theta_prim#\s*([\d\.eE\+\-]+)', line)
            if match:
                metadata['theta_prim'] = float(match.group(1))

            match = re.search(r'#phi_prim#\s*([\d\.eE\+\-]+)', line)
            if match:
                metadata['phi_prim'] = float(match.group(1))

            match = re.search(r'#dsn_events#\s*(\S+)', line)
            if match:
                metadata['dsn_events'] = match.group(1)

    return metadata

def pid_name(pid_raw):
    """Преобразует 'сырой' PID в читаемое имя"""
    base_pid = pid_raw // 1000
    return PID_MAP.get(base_pid, f"unknown({base_pid})")

def save_event_data(event_idx, particles, header, save_path):
    event_data = []

    for p in particles:
        raw_pid = int(p[0])
        base_pid = raw_pid // 1000
        name = pid_name(raw_pid)
        
        px, py, pz = p[1], p[2], p[3]
        x, y, t = p[4] / 100, p[5] / 100, p[6]
        energy = np.sqrt(px**2 + py**2 + pz**2)
        
        event_data.append({
            'event_id': event_idx,
            'particle_name': name,
            'pid': base_pid,
            'x': x,
            'y': y,
            't': t,
            'energy': energy,
        })
    
    df_event = pd.DataFrame(event_data)
    event_file = os.path.join(save_path, f"event_{event_idx}.csv")
    df_event.to_csv(event_file, index=False)


def parse_corsika_showers(file_path):
    """
    Читает бинарный файл CORSIKA и собирает для каждого события:
      - NeNKG и sNKG из блока EVTE
      - Энергию первичной частицы, θ и φ из блока EVTH
    Завершает при нахождении блока RUNE.
    Возвращает словарь params_dict, где ключ — event_id (от 0), значение — словарь параметров.
    """
    params_dict = {}
    current_shower = None
    tmp_theta = None
    tmp_phi = None
    tmp_E0 = None
    shower_count = 0

    with open(file_path, 'rb') as f:
        while True:
            marker = f.read(4)
            if len(marker) < 4:
                break
            record_size = struct.unpack('i', marker)[0]
            if record_size <= 0:
                print("Некорректный размер записи:", record_size)
                break

            block_bytes = f.read(record_size)
            end_marker = f.read(4)
            if len(block_bytes) != record_size or len(end_marker) < 4:
                print("Ошибка чтения или неверный завершающий маркер.")
                break

            block = np.frombuffer(block_bytes, dtype=np.float32)
            num_sub = 21
            sub_size = 273

            for j in range(num_sub):
                start = j * sub_size
                end = start + sub_size
                if end > len(block):
                    break

                sub = block[start:end]
                hdr = sub[0].tobytes()
                try:
                    tag = hdr.decode('ascii')
                except:
                    tag = ''

                if tag == 'EVTH':
                    shower_count += 1
                    current_shower = shower_count
                    tmp_E0 = sub[3]  

                    theta_rad = sub[10]
                    phi_rad = sub[11]
                    tmp_theta = np.degrees(theta_rad) 
                    tmp_phi = np.degrees(phi_rad)      

                elif tag == 'EVTE' and current_shower is not None:
                    i_ne = 175 + 10 - 1
                    i_s = 185 + 10 - 1
                    if len(sub) > max(i_ne, i_s):
                        ne_nkg = sub[i_ne]
                        s_nkg = sub[i_s]
                        params_dict[current_shower - 1] = {
                            'Ne': ne_nkg,
                            's': s_nkg,
                            'energy_prim': tmp_E0,
                            'theta_prim': tmp_theta,
                            'phi_prim': tmp_phi
                        }
                        current_shower = None
                    else:
                        print(f"Ливень {current_shower}: EVTE слишком короткий")

                elif tag == 'RUNE':
                    return params_dict

    return params_dict
