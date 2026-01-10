import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import json

# --- CONFIGURATION ---
INPUT_FILE = "simulated_parking_data.csv"
OUTPUT_MAP = "learned_parking_map.json"

# DBSCAN PARAMETERS (Crucial for "Auto-Learning")
# eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
# Since standard parking spots are ~5-6 meters long, a radius of 1.5m is tight enough to group a single car's positions.
EPSILON_RADIUS = 1.5 

# min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
# A car stays for 1 hour? That's 60 minutes of data points.
# An illegal parker stays for 2 mins? That's 2 data points.
# We set this to 10 to filter out the short-term illegal parkers.
MIN_SAMPLES = 10 

def train_model():
    print(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("Error: CSV file not found. Run the simulation first!")
        return

    # We only care about X and Y coordinates for clustering
    coords = df[['X', 'Y']].values

    print("Running DBSCAN Clustering (The 'Learning' Phase)...")
    # This is the "Magic" line that separates signal (parking) from noise (illegal)
    db = DBSCAN(eps=EPSILON_RADIUS, min_samples=MIN_SAMPLES).fit(coords)
    
    # The cluster labels (-1 means Noise/Illegal, 0, 1, 2... are valid spots)
    labels = db.labels_

    # --- STATISTICS ---
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print(f"\nTraining Complete!")
    print(f"Estimated number of Valid Parking Spots: {n_clusters_}")
    print(f"Number of noise points (Illegal/Traffic) removed: {n_noise_}")

    # --- EXTRACTING THE "LEARNED" MAP ---
    valid_spots = []
    
    # For every cluster found, calculate its center
    unique_labels = set(labels)
    for k in unique_labels:
        if k == -1:
            continue # Skip noise

        # Get all points belonging to this cluster
        class_member_mask = (labels == k)
        xy = coords[class_member_mask]
        
        # Calculate the Centroid (Geometric Center) of the spot
        centroid_x = np.mean(xy[:, 0])
        centroid_y = np.mean(xy[:, 1])
        
        valid_spots.append({
            "id": int(k),
            "x": float(centroid_x),
            "y": float(centroid_y),
            "status": "free" # Default state for the app
        })

    # Save to JSON
    with open(OUTPUT_MAP, "w") as f:
        json.dump(valid_spots, f, indent=4)
    print(f"Saved learned map to {OUTPUT_MAP}")

    # --- VISUALIZATION (Verify the Logic) ---
    plot_results(coords, labels, n_clusters_)

def plot_results(X, labels, n_clusters):
    # Black is noise (Illegal Parking), Colors are Valid Spots
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    plt.figure(figsize=(12, 4))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1] # Black for Noise

        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        
        # Plot the points
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6 if k != -1 else 3)

    plt.title(f'Learned Layout: {n_clusters} Valid Spots Found (Black Dots = Illegal/Noise)')
    plt.xlabel('Road Position X (meters)')
    plt.yticks([]) # Hide Y axis since it's a straight road
    plt.show()

if __name__ == "__main__":
    train_model()