import cv2
import matplotlib.patches as patches
import numpy as np
import glob
import os
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# Ensure this folder name matches exactly where your images are!
DATASET_PATH = "test/*.jpg" 

YOLO_MODEL = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.3

# --- TUNED PARAMETERS (CRITICAL FOR SUCCESS) ---
# Radius=15 separates the cars. Radius=40 merges them into blobs.
CLUSTERING_RADIUS = 15  
# Samples=10 ensures we find spots even if the test is short.
MIN_SAMPLES = 15  

def run_dataset_test():
    print("Loading YOLOv11 model...")
    model = YOLO(YOLO_MODEL)
    
    # Get list of images
    image_files = sorted(glob.glob(DATASET_PATH))
    
    if not image_files:
        print(f"Error: No images found at {DATASET_PATH}")
        print("Make sure the folder exists and contains .jpg files!")
        return

    print(f"Found {len(image_files)} images. Starting simulation...")
    
    long_term_memory = []
    
    # Create a window
    cv2.namedWindow("PKLot Dataset Simulation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("PKLot Dataset Simulation", 1024, 768)

    # Loop through the images
    for i, img_path in enumerate(image_files):
        frame = cv2.imread(img_path)
        if frame is None: continue

        # 1. DETECT
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, classes=[2, 7], verbose=False)
        
        current_centroids = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            # Save to memory
            long_term_memory.append([cx, cy])
            current_centroids.append((cx, cy))

        # 2. VISUALIZE LIVE
        for (cx, cy) in current_centroids:
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            
        cv2.putText(frame, f"Processing: {i+1}/{len(image_files)}", (30, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("PKLot Dataset Simulation", frame)
        
        # Press 'q' to stop early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    
    # --- AUTO-LEARN PHASE ---
    print("Dataset sequence finished.")
    if len(long_term_memory) > 0:
        analyze_data(long_term_memory, image_files[0]) 
    else:
        print("No cars detected. Check your images.")

def analyze_data(data_points, background_image_path):
    print("Running Auto-Learning Algorithm (Point-Based)...")
    data = np.array(data_points)
    
    # 1. DBSCAN CLUSTERING
    db = DBSCAN(eps=15, min_samples=10).fit(data)
    labels = db.labels_
    
    # Load background for plotting later
    bg = cv2.cvtColor(cv2.imread(background_image_path), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(bg) 
    
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    # --- STEP 1: FIND CANDIDATES & PLOT RAW DATA ---
    initial_candidates = []
    
    for k, col in zip(unique_labels, colors):
        if k == -1: continue 
        
        class_member_mask = (labels == k)
        xy = data[class_member_mask]
        
        # A. FILTER: Check Dimensions (Remove "Green Strips" / Traffic)
        x_min, y_min = np.min(xy, axis=0)
        x_max, y_max = np.max(xy, axis=0)
        width = x_max - x_min
        height = y_max - y_min
        
        if width > 60 or height > 60: continue # Too big (traffic)
        if len(xy) < 15: continue              # Too small (noise)

        # B. Calculate Centroid
        cx = np.mean(xy[:, 0])
        cy = np.mean(xy[:, 1])
        initial_candidates.append([cx, cy])
        
        # C. Plot the raw dots (Faintly) so you can see the data
        ax.scatter(xy[:, 0], xy[:, 1], s=5, c=[col], alpha=0.3)

    print(f"Found {len(initial_candidates)} candidate spots before merging.")

    # --- STEP 2: MERGE DUPLICATES (The Fix) ---
    final_spots = []
    MERGE_DISTANCE = 30  # Pixels
    
    for (new_x, new_y) in initial_candidates:
        is_duplicate = False
        for i, (exist_x, exist_y) in enumerate(final_spots):
            # Calculate distance
            dist = np.sqrt((new_x - exist_x)**2 + (new_y - exist_y)**2)
            if dist < MERGE_DISTANCE:
                # Merge them by averaging positions (optional, improves accuracy)
                final_spots[i] = ((exist_x + new_x)/2, (exist_y + new_y)/2)
                is_duplicate = True
                break
        
        if not is_duplicate:
            final_spots.append((new_x, new_y))
            
    print(f"CLEANUP: Reduced to {len(final_spots)} unique spots.")

    # --- STEP 3: VISUALIZE FINAL SPOTS ---
    for i, (cx, cy) in enumerate(final_spots):
        # Draw Green Circle
        circle = patches.Circle((cx, cy), radius=12, linewidth=2, 
                                edgecolor='#00FF00', facecolor='none')
        ax.add_patch(circle)
        
        # Draw Yellow Center Dot
        ax.scatter([cx], [cy], s=30, c='yellow', edgecolors='black', zorder=10)
        
        # Label ID
        ax.text(cx + 10, cy + 10, str(i), color='white', fontweight='bold', fontsize=8,
                 bbox=dict(facecolor='black', alpha=0.5, pad=1))

    ax.set_title(f"Learned Infrastructure: {len(final_spots)} Valid Spots (Merged & Filtered)")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_dataset_test()