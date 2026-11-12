"""Channel lineups. Always index EEG by channel name: THINGS caches, StarStim hardware and lab loaders use different orders."""

# StarStim 32 / NIC2 hardware order (LabRecorder XDF and .easy columns 0-31)
STARSTIM_32 = ["P8", "T8", "CP6", "FC6", "F8", "F4", "C4", "P4", "AF4", "Fp2", "Fp1", "AF3", "Fz", "FC2", "Cz", "CP2",
               "PO3", "O1", "Oz", "O2", "PO4", "Pz", "CP1", "FC1", "P3", "C3", "F3", "F7", "FC5", "CP5", "T7", "P7"]

# the 31 StarStim electrodes that also exist in THINGS-EEG2 (no Fz there): the old-lab and THINGS lab-contract channel set
STARSTIM_31 = [c for c in STARSTIM_32 if c != "Fz"]

# electrodes used for stimulation in the early memorability_stim sessions; excluded from every session analysis.
# The later optimal montage (FC2 Cz P8 O2) goes flat only in pulse runs; flat channels are interpolated per run.
EARLY_STIM_ELECTRODES = ("F3", "F4", "C3", "C4", "PO3", "PO4")
SESSION_26 = [c for c in STARSTIM_32 if c not in EARLY_STIM_ELECTRODES]
SHARED_25 = [c for c in SESSION_26 if c != "Fz"]          # in both the sessions and the old lab / THINGS 31-channel set

# occipital and parietal electrodes: the timing check matches on them, and the pooled rapid-stream decoder can use them alone
POSTERIOR_10 = ("O1", "Oz", "O2", "PO3", "PO4", "P7", "P8", "P3", "P4", "Pz")
