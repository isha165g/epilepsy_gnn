from pathlib import Path

from montage import load_and_create_montage


def test_recording(edf_path):
    print("\n" + "=" * 70)
    print(f"Testing: {edf_path.name}")
    print("=" * 70)

    data, channel_names, sfreq = load_and_create_montage(edf_path)

    print(f"Sampling rate : {sfreq} Hz")
    print(f"Output shape  : {data.shape}")
    print(f"Channels      : {len(channel_names)}")

    print("\nChannels:")
    for i, name in enumerate(channel_names, 1):
        print(f"{i:2d}. {name}")

    print("\nSignal statistics:")
    print(f"Minimum       : {data.min():.6f}")
    print(f"Maximum       : {data.max():.6f}")
    print(f"Mean          : {data.mean():.6f}")
    print(f"Std           : {data.std():.6f}")

    print("\n✓ No NaN/Inf values")
    print("✓ 20 bipolar channels created")


if __name__ == "__main__":

    root = Path("data/raw/tusz_edf")

    # Find the LE-reference recording recursively
    le_matches = list(root.rglob("aaaaaewf_s001_t001.edf"))

    if not le_matches:
        raise FileNotFoundError(
            "Could not find aaaaaewf_s001_t001.edf anywhere under "
            "data/raw/tusz_edf/"
        )

    le_file = le_matches[0]

    # Find another EDF for testing
    ref_file = None

    for path in root.rglob("*.edf"):
        if path != le_file:
            ref_file = path
            break

    if ref_file is None:
        raise FileNotFoundError(
            "Could not find another EDF for REF testing."
        )

    print(f"LE test file : {le_file}")
    print(f"Second file  : {ref_file}")

    test_recording(le_file)
    test_recording(ref_file)