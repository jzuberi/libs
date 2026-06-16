from itertools import chain
from typing import Iterable, List, Sequence

import networkx as nx


def flatten_list_of_lists(list_of_lists: Iterable[Sequence]) -> list:
    """
    Flatten a list of lists into a single list.
    """
    return list(chain(*list_of_lists))


def get_connected_components(edge_list: List[Sequence]) -> list[list[str]]:
    """
    Given an edge list, return connected components as lists of node IDs (strings).

    Args:
        edge_list: list of [u, v] pairs.

    Returns:
        List of components, each a list of node IDs (as strings), sorted descending.
    """
    ccomp_list: list[list[str]] = []
    G = nx.Graph()

    for e in edge_list:
        G.add_edge(str(e[0]), str(e[1]))

    components = nx.connected_components(G)

    for component in components:
        clist = list(component)
        clist = sorted(clist, reverse=True)
        ccomp_list.append(clist)

    return ccomp_list


def remove_indices(strings: list, remove_idxs: list[int]) -> list:
    """
    Remove items from a list by index.

    Args:
        strings: original list.
        remove_idxs: indices to remove.

    Returns:
        New list with those indices removed.
    """
    remove_set = set(remove_idxs)
    return [s for i, s in enumerate(strings) if i not in remove_set]
