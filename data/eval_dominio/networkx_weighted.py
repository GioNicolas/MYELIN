"""
Shortest path algorithms for weighted graphs.
"""

from collections import deque
from heapq import heappop, heappush
from itertools import count, islice

import networkx as nx
from networkx.algorithms.shortest_paths.generic import _build_paths_from_predecessors

__all__ = [
    "dijkstra_path",
    "dijkstra_path_length",
    "bidirectional_dijkstra",
    "single_source_dijkstra",
    "single_source_dijkstra_path",
    "single_source_dijkstra_path_length",
    "multi_source_dijkstra",
    "multi_source_dijkstra_path",
    "multi_source_dijkstra_path_length",
    "all_pairs_dijkstra",
    "all_pairs_dijkstra_path",
    "all_pairs_dijkstra_path_length",
    "dijkstra_predecessor_and_distance",
    "bellman_ford_path",
    "bellman_ford_path_length",
    "single_source_bellman_ford",
    "single_source_bellman_ford_path",
    "single_source_bellman_ford_path_length",
    "all_pairs_bellman_ford_path",
    "all_pairs_bellman_ford_path_length",
    "bellman_ford_predecessor_and_distance",
    "negative_edge_cycle",
    "find_negative_cycle",
    "goldberg_radzik",
    "johnson",
]


def _weight_function(G, weight):
    """Returns a function that returns the weight of an edge.

    The returned function is specifically suitable for input to
    functions :func:`_dijkstra` and :func:`_bellman_ford_relaxation`.

    Parameters
    ----------
    G : NetworkX graph.

    weight : string or function
        If it is callable, `weight` itself is returned. If it is a string,
        it is assumed to be the name of the edge attribute that represents
        the weight of an edge. In that case, a function is returned that
        gets the edge weight according to the specified edge attribute.

    Returns
    -------
    function
        This function returns a callable that accepts exactly three inputs:
        a node, an node adjacent to the first one, and the edge attribute
        dictionary for the eedge joining those nodes. That function returns
        a number representing the weight of an edge.

    If `G` is a multigraph, and `weight` is not callable, the
    minimum edge weight over all parallel edges is returned. If any edge
    does not have an attribute with key `weight`, it is assumed to
    have weight one.

    """
    if callable(weight):
        return weight
    # If the weight keyword argument is not callable, we assume it is a
    # string representing the edge attribute containing the weight of
    # the edge.
    if G.is_multigraph():
        return lambda u, v, d: min(attr.get(weight, 1) for attr in d.values())
    return lambda u, v, data: data.get(weight, 1)


@nx._dispatchable(edge_attrs="weight")
def dijkstra_path(G, source, target, weight="weight"):
    """Returns the shortest weighted path from source to target in G.

    Uses Dijkstra's Method to compute the shortest weighted path
    between two nodes in a graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node
        Starting node

    target : node
        Ending node

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number or None to indicate a hidden edge.

    Returns
    -------
    path : list
        List of nodes in a shortest path.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    NetworkXNoPath
        If no path exists between source and target.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> print(nx.dijkstra_path(G, 0, 4))
    [0, 1, 2, 3, 4]

    Find edges of shortest path in Multigraph

    >>> G = nx.MultiDiGraph()
    >>> G.add_weighted_edges_from([(1, 2, 0.75), (1, 2, 0.5), (2, 3, 0.5), (1, 3, 1.5)])
    >>> nodes = nx.dijkstra_path(G, 1, 3)
    >>> edges = nx.utils.pairwise(nodes)
    >>> list(
    ...     (u, v, min(G[u][v], key=lambda k: G[u][v][k].get("weight", 1)))
    ...     for u, v in edges
    ... )
    [(1, 2, 1), (2, 3, 0)]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    The weight function can be used to include node weights.

    >>> def func(u, v, d):
    ...     node_u_wt = G.nodes[u].get("node_weight", 1)
    ...     node_v_wt = G.nodes[v].get("node_weight", 1)
    ...     edge_wt = d.get("weight", 1)
    ...     return node_u_wt / 2 + node_v_wt / 2 + edge_wt

    In this example we take the average of start and end node
    weights of an edge and add it to the weight of the edge.

    The function :func:`single_source_dijkstra` computes both
    path and length-of-path if you need both, use that.

    See Also
    --------
    bidirectional_dijkstra
    bellman_ford_path
    single_source_dijkstra
    """
    (length, path) = single_source_dijkstra(G, source, target=target, weight=weight)
    return path


@nx._dispatchable(edge_attrs="weight")
def dijkstra_path_length(G, source, target, weight="weight"):
    """Returns the shortest weighted path length in G from source to target.

    Uses Dijkstra's Method to compute the shortest weighted path length
    between two nodes in a graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
        starting node for path

    target : node label
        ending node for path

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number or None to indicate a hidden edge.

    Returns
    -------
    length : number
        Shortest path length.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    NetworkXNoPath
        If no path exists between source and target.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> nx.dijkstra_path_length(G, 0, 4)
    4

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    The function :func:`single_source_dijkstra` computes both
    path and length-of-path if you need both, use that.

    See Also
    --------
    bidirectional_dijkstra
    bellman_ford_path_length
    single_source_dijkstra

    """
    if source not in G:
        raise nx.NodeNotFound(f"Node {source} not found in graph")
    if source == target:
        return 0
    weight = _weight_function(G, weight)
    length = _dijkstra(G, source, weight, target=target)
    try:
        return length[target]
    except KeyError as err:
        raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}") from err


@nx._dispatchable(edge_attrs="weight")
def single_source_dijkstra_path(G, source, cutoff=None, weight="weight"):
    """Find shortest weighted paths in G from a source node.

    Compute shortest path between source and all other reachable
    nodes for a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node
        Starting node for path.

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number or None to indicate a hidden edge.

    Returns
    -------
    paths : dictionary
        Dictionary of shortest path lengths keyed by target.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> path = nx.single_source_dijkstra_path(G, 0)
    >>> path[4]
    [0, 1, 2, 3, 4]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    See Also
    --------
    single_source_dijkstra, single_source_bellman_ford

    """
    return multi_source_dijkstra_path(G, {source}, cutoff=cutoff, weight=weight)


@nx._dispatchable(edge_attrs="weight")
def single_source_dijkstra_path_length(G, source, cutoff=None, weight="weight"):
    """Find shortest weighted path lengths in G from a source node.

    Compute the shortest path length between source and all other
    reachable nodes for a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
        Starting node for path

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number or None to indicate a hidden edge.

    Returns
    -------
    length : dict
        Dict keyed by node to shortest path length from source.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> length = nx.single_source_dijkstra_path_length(G, 0)
    >>> length[4]
    4
    >>> for node in [0, 1, 2, 3, 4]:
    ...     print(f"{node}: {length[node]}")
    0: 0
    1: 1
    2: 2
    3: 3
    4: 4

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    See Also
    --------
    single_source_dijkstra, single_source_bellman_ford_path_length

    """
    return multi_source_dijkstra_path_length(G, {source}, cutoff=cutoff, weight=weight)


@nx._dispatchable(edge_attrs="weight")
def single_source_dijkstra(G, source, target=None, cutoff=None, weight="weight"):
    """Find shortest weighted paths and lengths from a source node.

    Compute the shortest path length between source and all other
    reachable nodes for a weighted graph.

    Uses Dijkstra's algorithm to compute shortest paths and lengths
    between a source and all other reachable nodes in a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
        Starting node for path

    target : node label, optional
        Ending node for path

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.


    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number or None to indicate a hidden edge.

    Returns
    -------
    distance, path : pair of dictionaries, or numeric and list.
        If target is None, paths and lengths to all nodes are computed.
        The return value is a tuple of two dictionaries keyed by target nodes.
        The first dictionary stores distance to each target node.
        The second stores the path to each target node.
        If target is not None, returns a tuple (distance, path), where
        distance is the distance from source to target and path is a list
        representing the path from source to target.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> length, path = nx.single_source_dijkstra(G, 0)
    >>> length[4]
    4
    >>> for node in [0, 1, 2, 3, 4]:
    ...     print(f"{node}: {length[node]}")
    0: 0
    1: 1
    2: 2
    3: 3
    4: 4
    >>> path[4]
    [0, 1, 2, 3, 4]
    >>> length, path = nx.single_source_dijkstra(G, 0, 1)
    >>> length
    1
    >>> path
    [0, 1]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The weight function can be used to hide edges by returning None.
    So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
    will find the shortest red path.

    Based on the Python cookbook recipe (119466) at
    https://code.activestate.com/recipes/119466/

    This algorithm is not guaranteed to work if edge weights
    are negative or are floating point numbers
    (overflows and roundoff errors can cause problems).

    See Also
    --------
    single_source_dijkstra_path
    single_source_dijkstra_path_length
    single_source_bellman_ford
    """
    return multi_source_dijkstra(
        G, {source}, cutoff=cutoff, target=target, weight=weight
    )


@nx._dispatchable(edge_attrs="weight")
def multi_source_dijkstra_path(G, sources, cutoff=None, weight="weight"):
    """Find shortest weighted paths in G from a given set of source
    nodes.

    Compute shortest path between any of the source nodes and all other
    reachable nodes for a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    sources : non-empty set of nodes
        Starting nodes for paths. If this is just a set containing a
