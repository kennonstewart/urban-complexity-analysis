"""
Analyze a single city's road network.

This script accepts a city key (corresponding to a filename in data/raw),
loads the graph, and performs analysis including visualization, canonical
representations, and MDL statistics.
"""
import argparse
import gc
import math
import os
import random

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Point


def get_city_center(G):
    """Calculate the centroid of node coordinates."""
    xs = [data["x"] for _, data in G.nodes(data=True)]
    ys = [data["y"] for _, data in G.nodes(data=True)]
    return np.mean(xs), np.mean(ys)


def make_radii(max_km=10, step_km=0.5):
    """Generate array of radii in meters."""
    return np.arange(step_km * 1000, max_km * 1000 + 1, step_km * 1000)


def mdl_er_graph(G_sub):
    """Compute MDL for an Erdos-Renyi graph model."""
    n = G_sub.number_of_nodes()
    m = G_sub.number_of_edges()
    if n < 2:
        return 0.0

    N_pairs = n * (n - 1) / 2
    if N_pairs == 0:
        return 0.0

    p = m / N_pairs if N_pairs > 0 else 0.0

    # avoid log(0)
    eps = 1e-10
    p = min(max(p, eps), 1 - eps)

    logL = m * math.log(p) + (N_pairs - m) * math.log(1 - p)
    # convert to bits
    return -logL / math.log(2)


def compute_radius_subgraphs(G, center_lon, center_lat, radii_m):
    """Compute subgraphs at increasing radii from the city center."""
    # Convert nodes to projected CRS so distances are in meters
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

    # Project to a local metric CRS
    nodes_proj = nodes_gdf.to_crs(epsg=3857)  # Web Mercator is fine for experiment
    center_point = gpd.GeoSeries(
        [Point(center_lon, center_lat)], crs="EPSG:4326"
    ).to_crs(nodes_proj.crs)[0]

    # Precompute distances from center (in meters)
    distances = nodes_proj.geometry.distance(center_point)  # pandas Series indexed by node id

    radius_results = []

    for r in radii_m:
        # nodes within radius r
        nodes_in = distances[distances <= r].index
        if len(nodes_in) < 2:
            continue  # nothing interesting to encode yet

        G_r = G.subgraph(nodes_in).copy()
        radius_results.append((r, G_r))

    return radius_results


def mdl_stats(G_sub):
    """Compute MDL statistics for a subgraph."""
    n = G_sub.number_of_nodes()
    m = G_sub.number_of_edges()
    L = mdl_er_graph(G_sub)
    return {
        "n_nodes": n,
        "n_edges": m,
        "mdl_bits": L,
        "mdl_per_edge": L / m if m > 0 else float("nan"),
        "mdl_per_node": L / n if n > 0 else float("nan"),
    }


def plot_force_directed(G, place=None, sample_nodes=None, output_path=None):
    """Generate a force-directed layout visualization."""
    # Optionally restrict to a subset of nodes (for big graphs)
    if sample_nodes is not None and G.number_of_nodes() > sample_nodes:
        # simplest: take a random induced subgraph
        nodes = random.sample(list(G.nodes()), sample_nodes)
        H = G.subgraph(nodes).copy()
    else:
        H = G

    # Make sure it's undirected for layout stability
    H_und = H.to_undirected()

    # Compute spring layout (force-directed)
    pos = nx.spring_layout(
        H_und,
        seed=42,          # deterministic
        k=None,           # defaults based on 1/sqrt(n)
        iterations=100    # bump if layouts look messy
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#111111")

    nx.draw_networkx_edges(
        H_und,
        pos,
        ax=ax,
        width=0.5,
        alpha=1,
        edge_color="#69DDFF"
    )
    nx.draw_networkx_nodes(
        H_und,
        pos,
        ax=ax,
        node_size=8,
        node_color="#DBBADD",
        alpha=0.5,
    )

    if place is not None:
        ax.set_title(f"{place} – force-directed layout", color="white")

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  Saved force-directed plot to: {output_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze a city's road network")
    parser.add_argument(
        "--city",
        required=True,
        help="City key (filename in data/raw without extension)",
    )
    args = parser.parse_args()

    city_key = args.city

    # Set up paths
    base_dir = os.path.dirname(__file__)
    raw_dir = os.path.join(base_dir, "..", "data", "raw")
    results_dir = os.path.join(base_dir, "..", "data", "results")

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Load the graph
    graph_path = os.path.join(raw_dir, f"{city_key}.graphml")
    print(f"Loading graph from: {graph_path}")
    G = ox.load_graphml(graph_path)

    # Generate visualization (map)
    print("Generating network visualization...")
    map_path = os.path.join(results_dir, f"{city_key}_map.png")
    fig, ax = ox.plot_graph(
        G,
        bgcolor="#111111",
        node_color="#DBBADD",
        node_size=15,
        node_alpha=0.3,
        node_edgecolor="none",
        node_zorder=1,
        edge_color="#69DDFF",
        edge_alpha=0.3,
        show=False,
        close=False,
    )
    plt.savefig(map_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    plt.clf()
    gc.collect()
    print(f"  Saved map to: {map_path}")

    # Simplify and project the graph
    print("Simplifying and projecting graph...")
    try:
        G_simplified = ox.simplification.simplify_graph(G)
        G_proj = ox.project_graph(G_simplified)
        G_std = ox.convert.graph_to_gdfs(G_proj, nodes=True, edges=True)
    except Exception as e:
        print(f"  Warning: Simplification failed, using direct projection: {e}")
        G_proj = ox.project_graph(G)
        G_std = ox.convert.graph_to_gdfs(G_proj, nodes=True, edges=True)

    # Save canonical representations
    print("Saving canonical representations...")
    adjlist_path = os.path.join(results_dir, f"{city_key}.adjlist")
    g6_path = os.path.join(results_dir, f"{city_key}.g6")
    edgelist_path = os.path.join(results_dir, f"{city_key}.edgelist")

    nx.write_adjlist(G, adjlist_path)
    print(f"  Saved adjlist to: {adjlist_path}")

    g6 = nx.to_graph6_bytes(nx.Graph(G), header=False)
    with open(g6_path, "wb") as f:
        f.write(g6)
    print(f"  Saved g6 to: {g6_path}")

    nx.write_edgelist(G, edgelist_path, data=False)
    print(f"  Saved edgelist to: {edgelist_path}")

    # Cleanup intermediate objects
    if "G_simplified" in locals():
        del G_simplified
    del G_proj
    gc.collect()

    # Compute MDL statistics for radius subgraphs
    print("Computing MDL statistics for radius subgraphs...")
    center_lon, center_lat = get_city_center(G)
    radii_m = make_radii(max_km=10, step_km=0.5)

    radius_subgraphs = compute_radius_subgraphs(G, center_lon, center_lat, radii_m)

    radius_rows = []
    for r, G_r in radius_subgraphs:
        stats = mdl_stats(G_r)
        stats.update(
            {
                "place": city_key,
                "radius_m": r,
            }
        )
        radius_rows.append(stats)

    # Save MDL stats
    mdl_df = pd.DataFrame(radius_rows)
    mdl_stats_path = os.path.join(results_dir, f"{city_key}_mdl_stats.csv")
    mdl_df.to_csv(mdl_stats_path, index=False)
    print(f"  Saved MDL stats to: {mdl_stats_path}")

    # Cleanup subgraphs
    del radius_subgraphs
    gc.collect()

    # Generate force-directed plot
    print("Generating force-directed visualization...")
    force_directed_path = os.path.join(results_dir, f"{city_key}_force_directed.png")
    plot_force_directed(G, place=city_key, sample_nodes=1000, output_path=force_directed_path)

    print(f"Analysis complete for: {city_key}")


if __name__ == "__main__":
    main()
