import csv
import json
import os
from .db_connection import DatabaseConnection
from .config import MONGO_DB_NAME, OUTPUT_DIR

CLUSTER_COUNT = 9
STATIONS_PER_CLUSTER = 4
TOTAL_STATIONS = CLUSTER_COUNT * STATIONS_PER_CLUSTER


def bool_array(indices, size):
    arr = [False] * size
    for idx in indices:
        if 1 <= idx <= size:
            arr[idx-1] = True
    return arr


def list_run_collections(db):
    return [name for name in db.list_collection_names()
            if name.startswith("RUN_") and name.endswith("_events")]


def flatten_and_export():
    conn = DatabaseConnection()
    conn.add_database('events', MONGO_DB_NAME)
    db = conn.get_database('events')
    if db is None:
        raise RuntimeError("Не удалось подключиться к базе данных.")

    col_names = list_run_collections(db)

    header = [
        "NRUN", "NEvent", "NtrackX", "NtrackY", "Ntrack", "Theta", "Phi",
        "IdEv", "Nview",
        "clusters",      
        "stations",    
        "a_std",     
        "q_std",        
        "t_std"          
    ]

    out_file = os.path.join(OUTPUT_DIR, "events.csv")
    with open(out_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for col in col_names:
            for doc in db[col].find():
                run_num = doc.get("run")
                run_doc_key = f"_{run_num}_run_doc"
                base = doc.get(run_doc_key, {})

                row = [
                    base.get("NRUN"), base.get("NEvent"), base.get("NtrackX"),
                    base.get("NtrackY"), base.get("Ntrack"), base.get("Theta"),
                    base.get("Phi"), base.get("IdEv"), base.get("Nview")
                ]

                clusters = doc.get("data_events_doc", {}).get("list_of_cluster_numbers", [])
                cluster_flags = bool_array(clusters, CLUSTER_COUNT)
                station_flags = [False] * TOTAL_STATIONS
                a_std_vals = [None] * TOTAL_STATIONS
                q_std_vals = [None] * TOTAL_STATIONS
                t_std_vals = [None] * TOTAL_STATIONS

                for entry in doc.get("data_e_list", []):
                    cl = entry.get("cluster")
                    hits = entry.get("list_of_hit_ds_numbers", [])
                    sts = entry.get("stations", {})
                    for ds in hits:
                        idx = (cl - 1) * STATIONS_PER_CLUSTER + (ds - 1)
                        station_flags[idx] = True
                        st = sts.get(f"ds_{ds}", {})
                        a_std_vals[idx] = st.get("a_std")
                        q_std_vals[idx] = st.get("q_std")
                        t_std_vals[idx] = st.get("t_std")

                row += [
                    json.dumps(cluster_flags),
                    json.dumps(station_flags),
                    json.dumps(a_std_vals),
                    json.dumps(q_std_vals),
                    json.dumps(t_std_vals)
                ]

                writer.writerow(row)


