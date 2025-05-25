from __future__ import annotations

converters = {
    "AmplKSM": eval, "hSM": eval, "nTrackSMX": eval, "nTrackSMY": eval, "nTrackSM": eval,
    "hSM0": eval, "nTrackSMX0": eval, "nTrackSMY0": eval, "nTrackSM0": eval,
    "EdepCntSCT": eval, "EdepDetNE": eval, "TimDetNE": eval, "EdepStNE": eval, "TimStNE": eval
}

columns_to_eval = [
    "AmplKSM", "hSM", "nTrackSMX", "nTrackSMY", "nTrackSM",
    "hSM0", "nTrackSMX0", "nTrackSMY0", "nTrackSM0",
    "EdepCntSCT", "EdepDetNE", "TimDetNE", "EdepStNE", "TimStNE"
]