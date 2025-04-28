from pathlib import Path
import os
from modeling_parameters.config import READ_GEANT_OUTPUT_DIR, RECONSTRUCTION_GEANT_OUTPUT_DIR


input_path = READ_GEANT_OUTPUT_DIR
output_path = RECONSTRUCTION_GEANT_OUTPUT_DIR

converters={
        "AmplKSM": eval, "hSM": eval, "nTrackSMX": eval, "nTrackSMY": eval, "nTrackSM": eval,
        "hSM0": eval, "nTrackSMX0": eval, "nTrackSMY0": eval, "nTrackSM0": eval,
        "EdepCntSCT": eval, "EdepDetNE": eval, "TimDetNE": eval, "EdepStNE": eval, "TimStNE": eval
    }


bounds = [
    (-50, 50),
    (-80, 80),
    (1, 1e8),  
    (0.2, 2)  
]