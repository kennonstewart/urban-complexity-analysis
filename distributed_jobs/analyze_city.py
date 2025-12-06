"""
Analyze a single city's road network,
including generation of a geoJSON export of the network.

This script accepts a city key (corresponding to a filename in data/raw),
loads the graph, and performs analysis including visualization, canonical
representations, MDL statistics, and now GeoJSON export.

The analysis pipeline includes:
1. Loading the city's road network graph from GraphML format
2. Generating geographic network visualizations
3. Creating force-directed layout visualizations
4. Computing canonical graph representations (adjlist, g6, edgelist)
5. Computing MDL (Minimum Description Length) statistics at multiple radii
6. Saving all results to the data/results directory
7. Exporting the road network as GeoJSON (edges and nodes)

Usage:
    python analyze_city.py --city "Detroit, Michigan, USA"
"""
import argparse
import gc
import json
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
    xs = [data["x"] for _, data in G.nodes(data=True)]
    ys = [data["y"] for _, data in G.nodes(data=True)]
    center = (np.mean(xs), np.mean(ys))
    return center


def make_radii(max_km=10, step_km=0.5):
    radii = np.arange(step_km * 1000, max_km * 1000 + 1, step_km * 1000)
    return radii


def mdl_er_graph(G_sub):
    n = G_sub.number_of_nodes()
    m = G_sub.number_of_edges()
    if n < 2:
        return 0.0
    N_pairs = n * (n - 1) / 2
    if N_pairs == 0:
        return 0.0
    p = m / N_pairs if N_pairs > 0 else 0.0
    eps = 1e-10
    p = min(max(p, eps), 1 - eps)
    logL = m * math.log(p) + (N_pairs - m) * math.log(1 - p)
    mdl_bits = -logL / math.log(2)
    return mdl_bits


def compute_radius_subgraphs(G, center_lon, center_lat, radii_m):
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    nodes_proj = nodes_gdf.to_crs(epsg=3857)
    center_point = gpd.GeoSeries(
        [Point(center_lon, center_lat)], crs="EPSG:4326"
    ).to_crs(nodes_proj.crs)[0]
    distances = nodes_proj.geometry.distance(center_point)
    radius_results = []
    for r in radii_m:
        nodes_in = distances[distances <= r].index
        if len(nodes_in) < 2:
            continue
        G_r = G.subgraph(nodes_in).copy()
        radius_results.append((r, G_r))
    return radius_results


def mdl_stats(G_sub):
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
    return stats


def plot_force_directed(G, place=None, sample_nodes=None, output_path=None):
    if sample_nodes is not None and G.number_of_nodes() > sample_nodes:
        import random
        nodes = random.sample(list(G.nodes()), sample_nodes)
        H = G.subgraph(nodes).copy()
    else:
        H = G
    H_und = H.to_undirected()
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
    parser = argparse.ArgumentParser(
        description="Analyze a city's road network (includes GeoJSON export)",
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
    - {city}_edges.geojson: Road network (edges) as GeoJSON
    - {city}_nodes.geojson: Road network (nodes) as GeoJSON
        """
    )
    parser.add_argument(
        "--city",
        required=True,
        help="City key (filename in data/raw without extension)",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Only generate the 3D JSON export (skipping other analysis steps)",
    )
    args = parser.parse_args()
    city_key = args.city
    logger.info(f"="*60)
    logger.info(f"Starting analysis for: {city_key}")
    logger.info(f"="*60)
    base_dir = os.path.dirname(__file__)
    raw_dir = os.path.join(base_dir, "..", "data", "raw")
    results_dir = os.path.join(base_dir, "..", "data", "results")
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Raw data directory: {raw_dir}")
    logger.info(f"Results directory: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)
    logger.debug(f"Ensured results directory exists")
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
    if not args.json_only:
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
            nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_proj)
            logger.info("Successfully simplified and projected graph")
        except Exception as e:
            logger.warning(f"Simplification failed, using direct projection: {e}")
            G_proj = ox.project_graph(G)
            nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_proj)
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
    else:
        logger.info("Skipping steps 1-5 (visualization, canonical forms, MDL) due to --json-only")

    # Export to Plotly 3D JSON
    logger.info("Step 6/6: Exporting 3D graph JSON...")
    json_3d_path = os.path.join(results_dir, f"{city_key}_3d_graph.json")
    try:
        # Compute 3D force-directed layout
        logger.info("Computing 3D force-directed layout...")
        # Use the same graph G used for other steps
        # Convert to undirected for layout computation
        H_und = G.to_undirected()
        
        # Calculate layout
        # dim=3 for 3D layout
        pos_3d = nx.spring_layout(H_und, dim=3, seed=42, iterations=100)
        
        # Prepare node data
        node_x = []
        node_y = []
        node_z = []
        
        # We need to iterate over nodes in a consistent order
        for node_id in G.nodes():
            x, y, z = pos_3d[node_id]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)

        # Prepare edge traces (lines)
        edge_x = []
        edge_y = []
        edge_z = []
        
        # Iterate over edges
        for u, v in G.edges():
            if u in pos_3d and v in pos_3d:
                x0, y0, z0 = pos_3d[u]
                x1, y1, z1 = pos_3d[v]
                
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_z.extend([z0, z1, None])

        # Construct the Plotly JSON structure matching ann-arbor.json
        plotly_data = {
            "data": [
                {
                    "line": {"color": "lightblue", "width": 1},
                    "mode": "lines",
                    "x": edge_x,
                    "y": edge_y,
                    "z": edge_z,
                    "type": "scatter3d"
                },
                {
                    "mode": "markers",
                    "x": node_x,
                    "y": node_y,
                    "z": node_z,
                    "type": "scatter3d"
                }
            ],
            "layout": {
                "title": {"text": f"{city_key} force-directed 3D layout"},
                "showlegend": False
            }
        }
        
        with open(json_3d_path, 'w') as f:
            json.dump(plotly_data, f)
            
        logger.info(f"3D Graph JSON saved: {json_3d_path}")
        
    except Exception as e:
        logger.error(f"Error exporting 3D graph JSON: {e}")
        # Don't raise, just log error so other results are preserved

    logger.info("="*60)
    logger.info(f"Analysis complete for: {city_key}")
    logger.info(f"All results saved to: {results_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()