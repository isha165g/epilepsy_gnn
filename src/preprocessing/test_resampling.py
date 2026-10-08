from pathlib import Path

# pyrefly: ignore [missing-import]
import mne

from montage import load_and_create_montage


def test_file(path):
    print("\n" + "=" * 70)
    print(f"Testing: {path.name}")
    print("=" * 70)

    # Read original metadata
    raw = mne.io.read_raw_edf(
        path,
        preload=False,
        verbose="ERROR",
    )

    original_sfreq = raw.info["sfreq"]
    original_samples = raw.n_times

    print(f"Original sampling rate : {original_sfreq} Hz")
    print(f"Original samples       : {original_samples}")

    # Montage + resampling
    data, channels, new_sfreq = load_and_create_montage(
        path,
        target_sfreq=256,
    )

    print(f"New sampling rate      : {new_sfreq} Hz")
    print(f"New shape              : {data.shape}")
    print(f"New samples            : {data.shape[1]}")

    duration_original = original_samples / original_sfreq
    duration_new = data.shape[1] / new_sfreq

    print(f"Original duration      : {duration_original:.3f} sec")
    print(f"New duration           : {duration_new:.3f} sec")
    print(f"Duration difference    : "
          f"{abs(duration_original - duration_new):.6f} sec")

    assert data.shape[0] == 20
    assert new_sfreq == 256
    assert data.shape[1] > 0

    assert abs(duration_original - duration_new) < 0.1

    assert data.dtype.kind == "f"
    assert not __import__("numpy").isnan(data).any()
    assert not __import__("numpy").isinf(data).any()

    print("\n✓ 20 channels")
    print("✓ Resampled to 256 Hz")
    print("✓ Duration preserved")
    print("✓ No NaN/Inf")


if __name__ == "__main__":

    root = Path("data/raw/tusz_edf")

    # 250 Hz LE
    le_file = root / (
        "dev/aaaaaewf/s001_2006/02_tcp_le/"
        "aaaaaewf_s001_t001.edf"
    )

    # 256 Hz REF
    ref_file = root / (
        "eval/aaaaasip/s004_2015/01_tcp_ar/"
        "aaaaasip_s004_t000.edf"
    )

    # Find a 1000 Hz recording
    hz1000_file = None

    for path in root.rglob("*.edf"):
        raw = mne.io.read_raw_edf(
            path,
            preload=False,
            verbose="ERROR",
        )

        if raw.info["sfreq"] == 1000:
            hz1000_file = path
            break

    if hz1000_file is None:
        raise RuntimeError("No 1000 Hz recording found.")

    print("\nTesting 250 Hz recording...")
    test_file(le_file)

    print("\nTesting 256 Hz recording...")
    test_file(ref_file)

    print("\nTesting 1000 Hz recording...")
    test_file(hz1000_file)