from pathlib import Path

import numpy as np
import pandas as pd

from topology import build_shared_electrode_adjacency, CHANNELS


MANIFEST_PATH = Path(
    "data/metadata/normalized_feature_manifest.csv"
)

OUTPUT_DIR = Path(
    "data/graph"
)


def validate_graph_sample(
    x,
    adjacency,
    label,
):
    """Validate one graph sample."""

    # Node features
    assert x.shape == (20, 16), (
        f"Expected (20, 16), got {x.shape}"
    )

    assert x.dtype == np.float32, (
        f"Expected float32, got {x.dtype}"
    )

    assert np.all(np.isfinite(x)), (
        "NaN/Inf detected in node features"
    )

    # Graph
    assert adjacency.shape == (20, 20), (
        f"Expected adjacency (20,20), "
        f"got {adjacency.shape}"
    )

    assert np.allclose(
        adjacency,
        adjacency.T
    ), "Adjacency is not symmetric"

    assert np.all(
        np.diag(adjacency) == 0
    ), "Adjacency contains self-loops"

    assert label in (0, 1), (
        f"Invalid label: {label}"
    )


def main():

    print("=" * 70)
    print("BUILDING GRAPH DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    print(
        f"Manifest samples : "
        f"{len(manifest):,}"
    )

    # --------------------------------------------------------
    # Build fixed graph topology
    # --------------------------------------------------------

    adjacency = (
        build_shared_electrode_adjacency()
    )

    adjacency = adjacency.astype(
        np.float32
    )

    edge_count = int(
        np.sum(adjacency) // 2
    )

    print(
        f"Graph nodes       : "
        f"{len(CHANNELS)}"
    )

    print(
        f"Graph edges       : "
        f"{edge_count}"
    )

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    for split in [
        "train",
        "dev",
        "eval"
    ]:
        (
            OUTPUT_DIR / split
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Save topology once
    # --------------------------------------------------------

    np.save(
        OUTPUT_DIR / "adjacency.npy",
        adjacency
    )

    np.save(
        OUTPUT_DIR / "channels.npy",
        np.array(CHANNELS)
    )

    # --------------------------------------------------------
    # Build graph samples
    # --------------------------------------------------------

    graph_records = []

    for idx, row in manifest.iterrows():

        split = row["split"]

        feature_path = Path(
            row["normalized_feature_path"]
        )

        if not feature_path.exists():
            raise FileNotFoundError(
                f"Missing feature file:\n"
                f"{feature_path}"
            )

        # ----------------------------------------------------
        # Load normalized node features
        # ----------------------------------------------------

        x = np.load(
            feature_path
        )

        x = x.astype(
            np.float32
        )

        label = int(
            row["label"]
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        validate_graph_sample(
            x,
            adjacency,
            label
        )

        # ----------------------------------------------------
        # Create graph sample
        # ----------------------------------------------------

        graph_id = (
            f"{idx:06d}"
        )

        graph_path = (
            OUTPUT_DIR
            / split
            / f"{graph_id}.npz"
        )

        np.savez(
            graph_path,
            x=x,
            y=np.array(
                label,
                dtype=np.int64
            ),
            adjacency=adjacency
        )

        # ----------------------------------------------------
        # Store metadata
        # ----------------------------------------------------

        record = row.to_dict()

        record["graph_id"] = graph_id
        record["graph_path"] = str(
            graph_path
        )

        graph_records.append(
            record
        )

        if (idx + 1) % 500 == 0:
            print(
                f"Built "
                f"{idx + 1:,} / "
                f"{len(manifest):,}"
            )

    # --------------------------------------------------------
    # Save graph manifest
    # --------------------------------------------------------

    graph_manifest = pd.DataFrame(
        graph_records
    )

    graph_manifest_path = Path(
        "data/metadata/graph_manifest.csv"
    )

    graph_manifest.to_csv(
        graph_manifest_path,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GRAPH DATASET COMPLETE")
    print("=" * 70)

    print(
        f"Total graphs : "
        f"{len(graph_manifest):,}"
    )

    print("\nGraphs by split:")
    print(
        graph_manifest["split"]
        .value_counts()
        .sort_index()
    )

    print("\nGraphs by label:")
    print(
        graph_manifest["label"]
        .value_counts()
        .sort_index()
    )

    print(
        f"\nTopology:"
        f"\n  Nodes : {len(CHANNELS)}"
        f"\n  Edges : {edge_count}"
    )

    print(
        f"\nSaved topology:"
        f"\n{OUTPUT_DIR / 'adjacency.npy'}"
    )

    print(
        f"\nGraph manifest:"
        f"\n{graph_manifest_path}"
    )


if __name__ == "__main__":
    main()