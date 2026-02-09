import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

# --- CONFIGURATION ---
INPUT_FILE = "data/outputs/simulated_parking_data.csv"
OUTPUT_MAP = "data/outputs/learned_parking_map.json"

# Parking Slot Dimensions (Standard RTA/EU Size)
SLOT_LENGTH = 5.0  # meters
SLOT_WIDTH = 2.5   # meters

def generate_bounding_boxes():
    print(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("Error: CSV not found.")
        return

    coords = df[['X', 'Y']].values

    # 1. Run DBSCAN (The Learning Phase)
    print("Auto-learning parking layout...")
    db = DBSCAN(eps=1.5, min_samples=10).fit(coords)
    labels = db.labels_

    # 2. Extract Valid Spots
    unique_labels = set(labels)
    valid_spots = []
    
    # Setup the Plot
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_title(f"Generated Digital Twin: Infrastructure-Independent Map")
    ax.set_xlabel("Road Position (meters)")
    ax.set_ylabel("Lateral Position (meters)")
    
    # Plot the raw "Noise" (Illegal Parkers) as tiny black dots
    noise_mask = (labels == -1)
    ax.plot(coords[noise_mask, 0], coords[noise_mask, 1], 'k.', markersize=2, label='Noise / Illegal Parking')

    # 3. Draw Boxes for Valid Clusters
    spot_id_counter = 1
    for k in unique_labels:
        if k == -1: continue # Skip noise

        # Get all points for this spot
        class_member_mask = (labels == k)
        spot_coords = coords[class_member_mask]
        
        # Calculate Centroid
        cx = np.mean(spot_coords[:, 0])
        cy = np.mean(spot_coords[:, 1])
        
        # Calculate Bounding Box (Top-Left corner logic for Matplotlib)
        # Rectangle(xy, width, height) where xy is bottom-left
        box_x = cx - (SLOT_LENGTH / 2)
        box_y = cy - (SLOT_WIDTH / 2)
        
        # Add to List (for JSON)
        valid_spots.append({
            "id": spot_id_counter,
            "x": cx, "y": cy,
            "box": {
                "x": box_x, "y": box_y,
                "w": SLOT_LENGTH, "h": SLOT_WIDTH
            }
        })

        # DRAW THE BOX
        # Green border, semi-transparent green fill
        rect = patches.Rectangle(
            (box_x, box_y), SLOT_LENGTH, SLOT_WIDTH,
            linewidth=2, edgecolor='#10B981', facecolor='#10B98133'
        )
        ax.add_patch(rect)
        
        # Label the spot
        ax.text(cx, cy, str(spot_id_counter), color='black', 
                ha='center', va='center', fontweight='bold', fontsize=8)
        
        spot_id_counter += 1

    # Final visual adjustments
    ax.legend()
    ax.set_aspect('equal') # Important so boxes don't look stretched
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Show the plot
    plt.show()

    # Save the Learned Map
    with open(OUTPUT_MAP, "w") as f:
        json.dump(valid_spots, f, indent=4)
    print(f"Saved {len(valid_spots)} detected spots to {OUTPUT_MAP}")

if __name__ == "__main__":
    generate_bounding_boxes()