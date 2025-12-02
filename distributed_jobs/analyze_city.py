"""
Analyze a single city's road network.

This script accepts a city key (corresponding to a filename in data/raw),
loads the graph, and performs analysis including visualization, canonical
representations, and MDL statistics.

The analysis pipeline includes:
1. Loading the city's road network graph from GraphML format
2. Generating geographic network visualizations
3. Creating force-directed layout visualizations
4. Computing canonical graph representations (adjlist, g6, edgelist)
5. Computing MDL (Minimum Description Length) statistics at multiple radii
6. Saving all results to the data/results directory

Usage:
    python analyze_city.py --city "Detroit, Michigan, USA"
"""
import argparse
import gc
import logging
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_city_center(G):
    """Calculate the centroid of node coordinates.
    
    Args:
        G (networkx.MultiDiGraph): The road network graph with geographic coordinates.
        
    Returns:
        tuple: (longitude, latitude) of the network's centroid.
    """
    logger.debug(f"Computing city center from {G.number_of_nodes()} nodes")
    xs = [data["x"] for _, data in G.nodes(data=True)]
    ys = [data["y"] for _, data in G.nodes(data=True)]
    center = (np.mean(xs), np.mean(ys))
    logger.debug(f"City center: ({center[0]:.6f}, {center[1]:.6f})")
    return center


def make_radii(max_km=10, step_km=0.5):
    """Generate array of radii in meters.
    
    Args:
        max_km (float): Maximum radius in kilometers. Default is 10 km.
        step_km (float): Step size between radii in kilometers. Default is 0.5 km.
        
    Returns:
        numpy.ndarray: Array of radii values in meters.
    """
    radii = np.arange(step_km * 1000, max_km * 1000 + 1, step_km * 1000)
    logger.debug(f"Generated {len(radii)} radii from {step_km}km to {max_km}km")
    return radii


def mdl_er_graph(G_sub):
    """Compute MDL (Minimum Description Length) for an Erdos-Renyi graph model.
    
    The MDL principle provides a measure of the information needed to encode
    the graph structure under an Erdos-Renyi random graph assumption.
    
    Args:
        G_sub (networkx.Graph): The subgraph to analyze.
        
    Returns:
        float: MDL value in bits. Returns 0.0 for graphs with fewer than 2 nodes.
        
    Notes:
        - Uses the Erdos-Renyi model where each edge exists with probability p
        - p is estimated as the observed edge density: m / (n choose 2)
        - MDL = -log-likelihood converted to bits
    """
    n = G_sub.number_of_nodes()
    m = G_sub.number_of_edges()
    
    if n < 2:
        logger.debug(f"Graph too small for MDL (n={n}), returning 0.0")
        return 0.0

    N_pairs = n * (n - 1) / 2
    if N_pairs == 0:
        logger.warning("Zero possible edges, returning MDL=0.0")
        return 0.0

    p = m / N_pairs if N_pairs > 0 else 0.0

    # Avoid log(0) by clamping probability to safe range
    eps = 1e-10
    p = min(max(p, eps), 1 - eps)

    # Compute negative log-likelihood under ER model
    logL = m * math.log(p) + (N_pairs - m) * math.log(1 - p)
    # Convert to bits (log base 2)
    mdl_bits = -logL / math.log(2)
    
    logger.debug(f"MDL computation: n={n}, m={m}, p={p:.6f}, MDL={mdl_bits:.2f} bits")
    return mdl_bits


def compute_radius_subgraphs(G, center_lon, center_lat, radii_m):
    """Compute subgraphs at increasing radii from the city center.
    
    This function extracts nested subgraphs by selecting all nodes within
    a given distance from the city center. Distances are computed in meters
    using a projected coordinate system.
    
    Args:
        G (networkx.MultiDiGraph): The full road network graph.
        center_lon (float): Longitude of the city center.
        center_lat (float): Latitude of the city center.
        radii_m (numpy.ndarray): Array of radii in meters to extract subgraphs for.
        
    Returns:
        list: List of tuples (radius_meters, subgraph) for each radius that
              contains at least 2 nodes.
    """
    logger.info(f"Computing subgraphs for {len(radii_m)} radii")
    
    # Convert nodes to projected CRS so distances are in meters
    logger.debug("Converting graph to GeoDataFrames")
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

    # Project to a local metric CRS (Web Mercator - EPSG:3857)
    logger.debug("Projecting nodes to EPSG:3857 for distance calculations")
    nodes_proj = nodes_gdf.to_crs(epsg=3857)
    center_point = gpd.GeoSeries(
        [Point(center_lon, center_lat)], crs="EPSG:4326"
    ).to_crs(nodes_proj.crs)[0]

    # Precompute distances from center (in meters)
    logger.debug("Computing distances from center point")
    distances = nodes_proj.geometry.distance(center_point)  # pandas Series indexed by node id

    radius_results = []

    for r in radii_m:
        # Select nodes within radius r
        nodes_in = distances[distances <= r].index
        if len(nodes_in) < 2:
            logger.debug(f"Skipping radius {r}m - only {len(nodes_in)} node(s)")
            continue  # nothing interesting to encode yet

        G_r = G.subgraph(nodes_in).copy()
        logger.debug(f"Radius {r}m: {G_r.number_of_nodes()} nodes, {G_r.number_of_edges()} edges")
        radius_results.append((r, G_r))

    logger.info(f"Generated {len(radius_results)} subgraphs")
    return radius_results


def mdl_stats(G_sub):
    """Compute MDL statistics for a subgraph.
    
    Args:
        G_sub (networkx.Graph): The subgraph to analyze.
        
    Returns:
        dict: Dictionary containing:
            - n_nodes: Number of nodes
            - n_edges: Number of edges
            - mdl_bits: Total MDL in bits
            - mdl_per_edge: MDL normalized by edge count
            - mdl_per_node: MDL normalized by node count
    """
    n = G_sub.number_of_nodes()
    m = G_sub.number_of_edges()
    L = mdl_er_graph(G_sub)
    
    stats = {
        "n_nodes": n,
        "n_edges": m,
        "mdl_bits": L,
        "mdl_per_edge": L / m if m > 0 else float("nan"),
        "mdl_per_node": L / n if n > 0 else float("nan"),
    }
    
    logger.debug(f"MDL stats: {n} nodes, {m} edges, {L:.2f} bits")
    return stats


def plot_force_directed(G, place=None, sample_nodes=None, output_path=None):
    """Generate a force-directed layout visualization using notebook logic.

    Mirrors the simpler plotting approach from the analysis notebook:
    - Optional random node sampling for large graphs
    - Spring layout (force-directed) with fixed seed for determinism
    - Dark background styling with lower edge alpha (0.3)
    - Saves to PNG if an output path is provided

    Args:
        G (networkx.MultiDiGraph): Road network graph.
        place (str, optional): Title label (city name).
        sample_nodes (int, optional): Max nodes; randomly samples if exceeded.
        output_path (str, optional): File path to save PNG.
    """
    # Optionally restrict to a subset of nodes (for big graphs)
    if sample_nodes is not None and G.number_of_nodes() > sample_nodes:
        import random
        nodes = random.sample(list(G.nodes()), sample_nodes)
        H = G.subgraph(nodes).copy()
    else:
        H = G

    # Undirected for layout stability
    H_und = H.to_undirected()

    # Compute spring layout
    pos = nx.spring_layout(
        H_und,
        seed=42,
        k=None,
        iterations=100
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#111111")

    nx.draw_networkx_edges(
        H_und,
        pos,
        ax=ax,
        width=0.5,
        alpha=0.3,
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
        ax.set_title(f"{place} force-directed layout", color="white")

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved force-directed plot to: {output_path}")

    plt.close(fig)


def main():
    """Main execution function for city road network analysis.
    
    This function orchestrates the complete analysis pipeline:
    1. Loads the city's road network from GraphML
    2. Generates visualizations (map and force-directed layout)
    3. Exports canonical graph representations
    4. Computes MDL statistics at multiple spatial scales
    5. Saves all results to the data/results directory
    """
    parser = argparse.ArgumentParser(
        description="Analyze a city's road network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python analyze_city.py --city "Detroit, Michigan, USA"
    
Output files will be saved to data/results/:
    - {city}_map.png: Geographic network visualization
    - {city}_force_directed.png: Force-directed layout
    - {city}.adjlist: Adjacency list representation
    - {city}.g6: Graph6 canonical format
    - {city}.edgelist: Edge list representation
    - {city}_mdl_stats.csv: MDL statistics at multiple radii
        """
    )
    parser.add_argument(
        "--city",
        required=True,
        help="City key (filename in data/raw without extension)",
    )
    args = parser.parse_args()

    city_key = args.city
    logger.info(f"="*60)
    logger.info(f"Starting analysis for: {city_key}")
    logger.info(f"="*60)

    # Set up paths
    base_dir = os.path.dirname(__file__)
    # Go up one level to find the data directory
    raw_dir = os.path.join(base_dir, "..", "data", "raw")
    results_dir = os.path.join(base_dir, "..", "data", "results")
    
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Raw data directory: {raw_dir}")
    logger.info(f"Results directory: {results_dir}")

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    logger.debug(f"Ensured results directory exists")

    # Load the graph
    graph_path = os.path.join(raw_dir, f"{city_key}.graphml")
    logger.info(f"Loading graph from: {graph_path}")
    
    try:
        G = ox.load_graphml(graph_path)
        logger.info(f"Successfully loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except FileNotFoundError:
        logger.error(f"Graph file not found: {graph_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading graph: {e}")
        raise

    # Generate visualization (map)
    logger.info("Step 1/6: Generating geographic network visualization...")
    map_path = os.path.join(results_dir, f"{city_key}_map.png")
    try:
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
        logger.info(f"Saved map to: {map_path}")
    except Exception as e:
        logger.error(f"Error generating map visualization: {e}")
        raise

    # Simplify and project the graph
    logger.info("Step 2/6: Simplifying and projecting graph...")
    try:
        G_simplified = ox.simplification.simplify_graph(G)
        logger.debug(f"Simplified graph: {G_simplified.number_of_nodes()} nodes, {G_simplified.number_of_edges()} edges")
        G_proj = ox.project_graph(G_simplified)
        G_std = ox.convert.graph_to_gdfs(G_proj, nodes=True, edges=True)
        logger.info("Successfully simplified and projected graph")
    except Exception as e:
        logger.warning(f"Simplification failed, using direct projection: {e}")
        G_proj = ox.project_graph(G)
        G_std = ox.convert.graph_to_gdfs(G_proj, nodes=True, edges=True)
        logger.info("Successfully projected graph (without simplification)")

    # Save canonical representations
    logger.info("Step 3/6: Saving canonical graph representations...")
    adjlist_path = os.path.join(results_dir, f"{city_key}.adjlist")
    g6_path = os.path.join(results_dir, f"{city_key}.g6")
    edgelist_path = os.path.join(results_dir, f"{city_key}.edgelist")

    try:
        nx.write_adjlist(G, adjlist_path)
        logger.info(f"Saved adjlist to: {adjlist_path}")
    except Exception as e:
        logger.error(f"Error writing adjlist: {e}")
        raise

    try:
        g6 = nx.to_graph6_bytes(nx.Graph(G), header=False)
        with open(g6_path, "wb") as f:
            f.write(g6)
        logger.info(f"Saved g6 to: {g6_path}")
    except Exception as e:
        logger.error(f"Error writing g6 format: {e}")
        raise

    try:
        nx.write_edgelist(G, edgelist_path, data=False)
        logger.info(f"Saved edgelist to: {edgelist_path}")
    except Exception as e:
        logger.error(f"Error writing edgelist: {e}")
        raise

    # Cleanup intermediate objects
    logger.debug("Cleaning up intermediate graph objects")
    if "G_simplified" in locals():
        del G_simplified
    del G_proj
    gc.collect()

    # Compute MDL statistics for radius subgraphs
    logger.info("Step 4/6: Computing MDL statistics for radius subgraphs...")
    center_lon, center_lat = get_city_center(G)
    logger.info(f"City center coordinates: ({center_lon:.6f}, {center_lat:.6f})")
    radii_m = make_radii(max_km=10, step_km=0.5)
    logger.info(f"Analyzing {len(radii_m)} radii from 0.5km to 10km")

    radius_subgraphs = compute_radius_subgraphs(G, center_lon, center_lat, radii_m)

    logger.info(f"Computing MDL statistics for {len(radius_subgraphs)} valid subgraphs...")
    radius_rows = []
    for i, (r, G_r) in enumerate(radius_subgraphs, 1):
        logger.debug(f"Processing subgraph {i}/{len(radius_subgraphs)} (radius={r}m)")
        stats = mdl_stats(G_r)
        stats.update(
            {
                "place": city_key,
                "radius_m": r,
            }
        )
        radius_rows.append(stats)

    # Save MDL stats
    logger.info("Saving MDL statistics to CSV...")
    mdl_df = pd.DataFrame(radius_rows)
    mdl_stats_path = os.path.join(results_dir, f"{city_key}_mdl_stats.csv")
    mdl_df.to_csv(mdl_stats_path, index=False)
    logger.info(f"Saved MDL stats to: {mdl_stats_path}")
    logger.info(f"MDL stats summary: {len(radius_rows)} rows, radius range {radii_m[0]:.0f}m - {radii_m[-1]:.0f}m")

    # Cleanup subgraphs
    logger.debug("Cleaning up subgraph objects")
    del radius_subgraphs
    gc.collect()

    # Generate force-directed plot
    logger.info("Step 5/6: Generating force-directed visualization...")
    force_directed_path = os.path.join(results_dir, f"{city_key}_force_directed.png")
    try:
        plot_force_directed(G, place=city_key, output_path=force_directed_path)
    except Exception as e:
        logger.error(f"Error generating force-directed plot: {e}")
        raise

    logger.info("="*60)
    logger.info(f"Analysis complete for: {city_key}")
    logger.info(f"All results saved to: {results_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
