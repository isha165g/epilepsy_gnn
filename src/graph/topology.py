import numpy as np


# ============================================================
# 20 standardized bipolar EEG channels
# ============================================================

CHANNELS = [
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",

    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",

    "T3-C3",
    "C3-CZ",
    "CZ-C4",
    "C4-T4",

    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",

    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
]


# ============================================================
# Helper functions
# ============================================================

def get_electrodes(channel):
    """
    Convert a bipolar channel name into its two electrodes.

    Example:
        FP1-F7 -> ("FP1", "F7")
    """
    left, right = channel.split("-")
    return left, right


def channels_share_electrode(channel_a, channel_b):
    """
    Return True if two bipolar channels share an electrode.
    """
    electrodes_a = set(get_electrodes(channel_a))
    electrodes_b = set(get_electrodes(channel_b))

    return len(electrodes_a.intersection(electrodes_b)) > 0


# ============================================================
# Build static topology
# ============================================================

def build_shared_electrode_adjacency():
    """
    Build a 20 x 20 binary adjacency matrix.

    Two nodes are connected if their bipolar derivations
    share an underlying electrode.

    Diagonal is kept at zero.
    """

    num_channels = len(CHANNELS)

    adjacency = np.zeros(
        (num_channels, num_channels),
        dtype=np.float32
    )

    for i in range(num_channels):
        for j in range(i + 1, num_channels):

            if channels_share_electrode(
                CHANNELS[i],
                CHANNELS[j]
            ):
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0

    return adjacency


def get_edge_list(adjacency):
    """
    Convert an adjacency matrix into an undirected edge list.

    Each edge is returned only once.

    Example:
        [(0, 1), (1, 2), ...]
    """

    edges = []

    num_nodes = adjacency.shape[0]

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):

            if adjacency[i, j] > 0:
                edges.append((i, j))

    return edges


def print_topology(adjacency):
    """
    Print the graph in a human-readable form.
    """

    edges = get_edge_list(adjacency)

    print("\nGRAPH TOPOLOGY")
    print("=" * 60)

    print(f"Number of nodes : {len(CHANNELS)}")
    print(f"Number of edges : {len(edges)}")

    print("\nNodes:")
    for i, channel in enumerate(CHANNELS):
        print(f"{i:2d}: {channel}")

    print("\nEdges:")
    for i, j in edges:
        print(
            f"{i:2d} ({CHANNELS[i]})"
            f"  <->  "
            f"{j:2d} ({CHANNELS[j]})"
        )


if __name__ == "__main__":

    adjacency = build_shared_electrode_adjacency()

    print_topology(adjacency)

    print("\nAdjacency matrix:")
    print(adjacency)