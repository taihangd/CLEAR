"""Minimum travel-time evidence within the supplied road-network model; not equivalent to full map matching or topology verification.

Checked only when output points uniquely associate with graph nodes within a strict spatial tolerance.
Edge weights and directions are interpreted from the complete caller-supplied graph; missing paths,
ambiguous node associations, or invalid metadata are reported as unknown.
Never reads GT, modifies trajectories, mutates the input graph, or guesses missing edge lengths.
"""

from collections import OrderedDict
import hashlib
import json
import math
import numbers

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree


_SCOPE = "supplied_graph_conditional_node_association"
_BASIS = (
    "Minimum weighted distance in the supplied graph, conditional on unique "
    "endpoint node association and declared planar distance/time units. "
    "This does not verify map completeness, target identity or final-trajectory map matching."
)


def _number(value, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError("non_numeric_value")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError("invalid_numeric_value")
    return value


def _node_value(node):
    """Stable, type-preserving digest encoding that avoids memory-address-dependent repr strings."""
    if isinstance(node, str):
        return ["str", node]
    if isinstance(node, (bool, np.bool_)):
        return ["bool", bool(node)]
    if isinstance(node, numbers.Integral):
        return ["int", int(node)]
    if isinstance(node, numbers.Real) and math.isfinite(float(node)):
        return ["float", float(node)]
    if isinstance(node, tuple):
        return ["tuple", [_node_value(part) for part in node]]
    raise ValueError("unsupported_node_or_edge_key_type")


def _node_label(node):
    value = _node_value(node)
    return value[1] if value[0] != "tuple" else value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def edge_travel_directions(graph, source, target, data):
    """Unify legal travel directions across prompts, checkers, and programmatic repairs without guessing ambiguous or conflicting metadata."""
    oneway = data.get("oneway")
    if oneway is not None and not isinstance(oneway, (bool, np.bool_)):
        raise ValueError("ambiguous_oneway_metadata")
    if not graph.is_directed() and oneway is not None and bool(oneway):
        raise ValueError("conflicting_undirected_oneway_metadata")
    directions = ((source, target),)
    if source != target and (not graph.is_directed() or (oneway is not None and not bool(oneway))):
        directions += ((target, source),)
    return directions


def prepare_road_time_graph(graph, spatial_scale=1.0, mapping_tolerance_m=0.001):
    """Copy and normalize edge weights; source shortest-path cache is scoped to this prepared object and bounded to 32 sources.

    Edge lengths and node coordinates share the same explicit spatial unit. A DiGraph follows its edge orientation
    when oneway is omitted and adds a reverse edge only when oneway is explicitly False. An undirected graph represents
    bidirectional travel and raises a metadata conflict if any edge declares oneway=True. Parallel edges keep the minimum weight while all raw edges enter the digest.
    """
    prepared = {"scope": _SCOPE, "hash": None}
    try:
        scale = _number(spatial_scale, positive=True)
        tolerance = _number(mapping_tolerance_m)
        if not isinstance(graph, nx.Graph) or not graph.number_of_nodes():
            raise ValueError("missing_or_empty_road_graph")
        normalized = nx.DiGraph()
        nodes, coordinates, node_records, edge_records = [], [], [], []
        for node, data in graph.nodes(data=True):
            coordinate = [float(data[axis]) for axis in ("x", "y")]
            if not all(math.isfinite(value) for value in coordinate):
                raise ValueError("nonfinite_node_coordinates")
            if any(isinstance(data[axis], (bool, np.bool_)) for axis in ("x", "y")):
                raise ValueError("boolean_node_coordinates")
            nodes.append(node)
            coordinates.append(coordinate)
            node_records.append([_node_value(node), *coordinate])
            normalized.add_node(node, x=coordinate[0], y=coordinate[1])

        def add_edge(source, target, weight):
            old = normalized.get_edge_data(source, target)
            if old is None or weight < old["weight"]:
                normalized.add_edge(source, target, weight=weight)

        edges = graph.edges(keys=True, data=True) if graph.is_multigraph() else (
            (source, target, None, data) for source, target, data in graph.edges(data=True))
        for source, target, key, data in edges:
            weight = _number(data["length"]) * scale
            if not math.isfinite(weight):
                raise ValueError("nonfinite_scaled_length")
            oneway = data.get("oneway")
            directions = edge_travel_directions(graph, source, target, data)
            source_value, target_value = _node_value(source), _node_value(target)
            if not graph.is_directed() and _canonical(source_value) > _canonical(target_value):
                source_value, target_value = target_value, source_value
            edge_records.append([
                source_value, target_value, None if key is None else _node_value(key),
                weight, None if oneway is None else bool(oneway),
            ])
            for left, right in directions:
                add_edge(left, right, weight)
        coordinates = np.asarray(coordinates, dtype=float) * scale
        if not np.isfinite(coordinates).all():
            raise ValueError("nonfinite_scaled_coordinates")
        fingerprint = {
            "version": 1, "directed": bool(graph.is_directed()), "multigraph": bool(graph.is_multigraph()),
            "nodes": sorted(node_records, key=_canonical), "edges": sorted(edge_records, key=_canonical),
            "spatial_scale": scale, "mapping_tolerance_m": tolerance,
        }
        prepared.update({
            "hash": hashlib.sha256(_canonical(fingerprint).encode()).hexdigest(),
            "graph": normalized, "nodes": nodes, "coordinates_m": coordinates,
            "tree": cKDTree(coordinates), "spatial_scale": scale, "mapping_tolerance_m": tolerance,
            "source_cache": OrderedDict(), "source_cache_limit": 32,
            "node_count": len(nodes), "edge_count": normalized.number_of_edges(),
            "input_directed": bool(graph.is_directed()),
        })
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, nx.NetworkXException) as error:
        # Do not expose object reprs, file paths, or raw exception messages in model feedback.
        prepared["error"] = "unusable_road_graph_metadata"
        prepared["error_type"] = type(error).__name__
    return prepared
