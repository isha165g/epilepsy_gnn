import numpy as np
# pyrefly: ignore [missing-import]
import mne

from montage import create_bipolar_montage


EDF = (
    "data/raw/tusz_edf/dev/aaaaaewf/"
    "s001_2006/02_tcp_le/"
    "aaaaaewf_s001_t001.edf"
)


raw = mne.io.read_raw_edf(
    EDF,
    preload=False,
    verbose="ERROR",
)

data, channels = create_bipolar_montage(raw)

sfreq = raw.info["sfreq"]

print("=" * 80)
print("CHECKING EEG SIGNAL START")
print("=" * 80)

print(f"\nSampling rate: {sfreq} Hz")
print(f"Duration     : {raw.times[-1]:.2f} sec")
print(f"Channels     : {data.shape[0]}")

# Check consecutive 5-second blocks
block_size = int(5 * sfreq)

print("\n5-second blocks:")

for i in range(12):

    start = i * block_size
    end = start + block_size

    block = data[:, start:end]

    if block.shape[1] == 0:
        break

    nonzero = np.count_nonzero(block)
    std = block.std()
    maximum = np.max(np.abs(block))

    print(
        f"  {i:02d}: "
        f"{i*5:3d}-{(i+1)*5:3d}s | "
        f"nonzero={nonzero:7d} | "
        f"std={std:.3e} | "
        f"max={maximum:.3e}"
    )

# Find first non-zero sample across any bipolar channel
nonzero_columns = np.any(data != 0, axis=0)

if np.any(nonzero_columns):

    first_sample = np.argmax(nonzero_columns)
    first_time = first_sample / sfreq

    print("\nFirst non-zero sample:")
    print(f"  sample : {first_sample}")
    print(f"  time   : {first_time:.3f} sec")

else:
    print("\nERROR: Entire recording is zero!")

print("\n" + "=" * 80)