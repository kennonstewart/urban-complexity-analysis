import argparse
import os
import networkx as nx
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def load_graph(city_key):
    """
    Load the city graph from edgelist file.
    """
    edgelist_path = os.path.join(DATA_DIR, f"{city_key}.edgelist")
    if not os.path.exists(edgelist_path):
        raise FileNotFoundError(f"Edgelist file not found for city: {city_key}")
    G = nx.read_edgelist(edgelist_path)
    return G


def generate_visual(city_key):
    """
    Generate and save a visualization for the given city graph.
    """
    G = load_graph(city_key)
    plt.figure(figsize=(12, 12))
    nx.draw(G, node_size=10, edge_color="#888", with_labels=False)
    output_path = os.path.join(RESULTS_DIR, f"{city_key}_visual.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved visual for {city_key} to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate city road network visualizations.")
    parser.add_argument('--city', required=True, help='City key (e.g., "Detroit, Michigan, USA")')
    args = parser.parse_args()
    city_key = args.city
    try:
        generate_visual(city_key)
    except Exception as e:
        print(f"Error generating visual for {city_key}: {e}")
        exit(1)


if __name__ == "__main__":
    main()
