import numpy as np

from topology import (
    CHANNELS,
    build_shared_electrode_adjacency,
    get_edge_list,
)


def test_node_count(adjacency):
    assert len(CHANNELS) == 20
    assert adjacency.shape == (20, 20)


def test_no_self_loops(adjacency):
    assert np.all(np.diag(adjacency) == 0)


def test_symmetric(adjacency):
    assert np.array_equal(adjacency, adjacency.T)


def test_binary(adjacency):
    assert np.all(
        np.isin(adjacency, [0.0, 1.0])
    )


def test_expected_edges(adjacency):
    expected_edges = [
        ("FP1-F7", "F7-T3"),
        ("F7-T3", "T3-T5"),
        ("F7-T3", "T3-C3"),
        ("T3-C3", "C3-CZ"),
        ("T3-C3", "F3-C3"),
        ("T3-C3", "C3-P3"),
        ("C3-CZ", "CZ-C4"),
        ("CZ-C4", "C4-T4"),
        ("C4-T4", "T4-T6"),
        ("C4-T4", "F8-T4"),
    ]

    channel_to_index = {
        channel: i
        for i, channel in enumerate(CHANNELS)
    }

    for channel_a, channel_b in expected_edges:

        i = channel_to_index[channel_a]
        j = channel_to_index[channel_b]

        assert adjacency[i, j] == 1
        assert adjacency[j, i] == 1


def test_edge_count(adjacency):
    edges = get_edge_list(adjacency)

    assert len(edges) > 0


def test_connected(adjacency):
    """
    Verify that every node can be reached from node 0.
    """

    visited = set()
    stack = [0]

    while stack:

        node = stack.pop()

        if node in visited:
            continue

        visited.add(node)

        neighbors = np.where(
            adjacency[node] > 0
        )[0]

        stack.extend(neighbors.tolist())

    assert len(visited) == 20


def main():

    adjacency = build_shared_electrode_adjacency()

    test_node_count(adjacency)
    test_no_self_loops(adjacency)
    test_symmetric(adjacency)
    test_binary(adjacency)
    test_expected_edges(adjacency)
    test_edge_count(adjacency)
    test_connected(adjacency)

    print("\n" + "=" * 60)
    print("ALL TOPOLOGY TESTS PASSED")
    print("=" * 60)

    print(f"Nodes : {len(CHANNELS)}")
    print(
        f"Edges : "
        f"{len(get_edge_list(adjacency))}"
    )


if __name__ == "__main__":
    main()