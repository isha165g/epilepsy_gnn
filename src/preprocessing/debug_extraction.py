import numpy as np
# pyrefly: ignore [missing-import]
import mne

from montage import create_bipolar_montage, resample_data
from filtering import bandpass_filter, notch_filter


EDF = (
    "data/raw/tusz_edf/dev/aaaaaewf/"
    "s001_2006/02_tcp_le/"
    "aaaaaewf_s001_t001.edf"
)


def stats(name, data):
    print(f"\n{name}")
    print(f"  shape : {data.shape}")
    print(f"  min   : {data.min():.12e}")
    print(f"  max   : {data.max():.12e}")
    print(f"  mean  : {data.mean():.12e}")
    print(f"  std   : {data.std():.12e}")
    print(f"  nonzero: {np.count_nonzero(data)}")


print("=" * 80)
print("DEBUGGING EEG EXTRACTION")
print("=" * 80)

# ---------------------------------------------------------
# 1. Load EDF
# ---------------------------------------------------------

raw = mne.io.read_raw_edf(
    EDF,
    preload=False,
    verbose="ERROR",
)

print("\nEDF")
print(f"  sampling rate : {raw.info['sfreq']}")
print(f"  duration      : {raw.times[-1]:.2f} sec")
print(f"  channels      : {len(raw.ch_names)}")

# ---------------------------------------------------------
# 2. Create bipolar montage
# ---------------------------------------------------------

data, channels = create_bipolar_montage(raw)

stats("Native bipolar EEG", data)

# ---------------------------------------------------------
# 3. Take first 5 seconds
# ---------------------------------------------------------

sfreq = raw.info["sfreq"]

end_sample = int(round(5 * sfreq))

segment = data[:, :end_sample]

stats("Native 0-5 sec", segment)

# ---------------------------------------------------------
# 4. Resample
# ---------------------------------------------------------

segment_resampled = resample_data(
    segment,
    original_sfreq=sfreq,
    target_sfreq=256,
)

stats("Resampled 0-5 sec", segment_resampled)

# ---------------------------------------------------------
# 5. Bandpass only
# ---------------------------------------------------------

bandpassed = bandpass_filter(
    segment_resampled,
    sfreq=256,
)

stats("After bandpass", bandpassed)

# ---------------------------------------------------------
# 6. Notch
# ---------------------------------------------------------

notched = notch_filter(
    bandpassed,
    sfreq=256,
)

stats("After notch", notched)

print("\n" + "=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)