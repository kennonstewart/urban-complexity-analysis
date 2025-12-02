"""
Download city road networks sequentially with API rate limiting.

This script downloads graph data for multiple cities and saves them
as GraphML files. It respects API limits by sleeping between requests.
"""
import os
import time
import osmnx as ox


def sanitize_name(place):
    """Convert place name to a filesystem-safe format."""
    return place.replace(" ", "_").replace(",", "_")


def main():
    # Define the list of places to download
    places = [
        "Manhattan, New York, USA",
        "San Francisco, California, USA",
        "Detroit, Michigan, USA",
        "Seattle, Washington, USA",
    ]

    # Ensure the data/raw directory exists
    raw_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Path to manifest file
    manifest_path = os.path.join(os.path.dirname(__file__), "data", "manifest.txt")

    # Clear the manifest file if it exists
    open(manifest_path, "w").close()

    # Download each city's graph
    for i, place in enumerate(places):
        print(f"Downloading graph for: {place}")

        # Download the graph
        G = ox.graph_from_place(place, network_type="drive")

        # Create sanitized filename
        sanitized_name = sanitize_name(place)
        output_path = os.path.join(raw_dir, f"{sanitized_name}.graphml")

        # Save the graph
        ox.save_graphml(G, output_path)
        print(f"  Saved to: {output_path}")

        # Append to manifest
        with open(manifest_path, "a") as f:
            f.write(f"{sanitized_name}\n")

        # Sleep between requests to respect API limits (except for the last one)
        if i < len(places) - 1:
            print("  Sleeping for 5 seconds...")
            time.sleep(5)

    print(f"Download complete. Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
