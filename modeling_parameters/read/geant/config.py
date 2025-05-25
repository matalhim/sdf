from __future__ import annotations
prim_particle = 'spe'
theta = 27

columns = [
    "N_event", "NRUN", "NEVENT", "PART0", "E0", "Teta", "Fi", "XAxisShift", "YAxisShift", "H1INT",
    "NGAM", "NEL", "NHADR", "NMU", "NeNKGlong", "sNKGlong", "NVD_edep", "NVD_npe", "MuBundle", "MuTrackLenNVD",
    "nMuNVD", "eMuNVD", "eMuNVD1", "muDCR", "muSM", "nSM", "muDCRw", "muSMw", "nSMw",
    "muDCR0", "muSM0", "nSM0", "muDCRw0", "muSMw0", "nSMw0",
    "AmplKSM", "hSM", "nTrackSMX", "nTrackSMY", "nTrackSM", "hSM0", "nTrackSMX0", "nTrackSMY0", "nTrackSM0",
    "EdepCntSCT", "EdepDetNE", "TimDetNE", "EdepStNE", "TimStNE", "marker"
]

sizes = {
    "AmplKSM": (7, 4, 4, 6),
    "hSM": (8,),
    "nTrackSMX": (8,),
    "nTrackSMY": (8,),
    "nTrackSM": (8,),
    "hSM0": (8,),
    "nTrackSMX0": (8,),
    "nTrackSMY0": (8,),
    "nTrackSM0": (8,),
    "EdepCntSCT": (9, 5, 2),
    "EdepDetNE": (9, 4, 4),
    "TimDetNE": (9, 4, 4, 4),
    "EdepStNE": (9, 4),
    "TimStNE": (9, 4, 4),
}