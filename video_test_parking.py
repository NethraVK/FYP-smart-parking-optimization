import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- CONFIGURATION ---
VIDEO_PATH = "BLK-HDPTZ12 Security Camera Parkng Lot Surveillance Video.mp4"  # <--- RENAME THIS to your video file
YOLO_MODEL = "yolo11n.pt"            # Will download automatically
CONFIDENCE_THRESHOLD = 0.3           # Lower confidence allowed for difficult shadows
PROCESS_EVERY_N_FRAMES = 10          # Skip frames to speed up processing

# PARKING SPOT ESTIMATION (Meters to Pixels ratio)
# You might need to tweak this EPSILON based on your video resolution
# If spots are merging too much, lower this. If spots are fragmented, raise it.
CLUSTERING_RADIUS_PIXELS = 50 
MIN_SAMPLES_TO_BE_A_SPOT = 15     # A car must be seen in ~15 frames to count

def process_video():
    print(f"Loading YOLOv11 model ({YOLO_MODEL})...")
    model = YOLO(YOLO_MODEL)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return

    # To draw the final map, we need a reference frame (the first one)
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Video is empty")
        return
    
    # Convert BGR to RGB for Matplotlib later
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    print("Processing video to extract vehicle positions...")
    
    all_vehicle_centroids = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Skip frames for speed
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            continue
            
        # Run YOLO Inference
        # classes=[2, 7] means only detect cars (2) and trucks (7)
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, classes=[2, 7], verbose=False)
        
        for box in results[0].boxes:
            # Get coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Calculate Centroid
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            all_vehicle_centroids.append([cx, cy])

        # Progress indicator
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames... Found {len(all_vehicle_centroids)} points.")

    cap.release()
    print("Video processing complete.")
    
    # --- PHASE 2: AUTO-LEARNING (DBSCAN) ---
    if len(all_vehicle_centroids) == 0:
        print("No vehicles detected! Check your video path or model.")
        return

    print("Running Auto-Learning Algorithm (DBSCAN)...")
    data = np.array(all_vehicle_centroids)
    
    # CLUSTER THE DOTS
    db = DBSCAN(eps=CLUSTERING_RADIUS_PIXELS, min_samples=MIN_SAMPLES_TO_BE_A_SPOT).fit(data)
    labels = db.labels_
    
    # Count valid spots (ignoring -1 noise)
    n_spots = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Auto-Learning discovered {n_spots} parking spots based on vehicle behavior.")

    # --- PHASE 3: VISUALIZATION ---
    visualize_results(first_frame_rgb, data, labels, n_spots)

def visualize_results(background_img, data, labels, n_spots):
    # Setup the plot using the video frame as background
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(background_img)
    
    unique_labels = set(labels)
    
    # Plot formatting
    ax.set_title(f"Validation Result: {n_spots} Spots Auto-Detected from Video")
    ax.axis('off') # Hide axes for cleaner look

    spot_id = 1
    for k in unique_labels:
        if k == -1:
            # Optional: Plot noise (moving cars) as tiny red dots
            # class_member_mask = (labels == k)
            # xy = data[class_member_mask]
            # ax.plot(xy[:, 0], xy[:, 1], 'r.', markersize=1, alpha=0.3)
            continue

        # Get all points for this spot
        class_member_mask = (labels == k)
        xy = data[class_member_mask]
        
        # 1. Find the center
        cx = np.mean(xy[:, 0])
        cy = np.mean(xy[:, 1])
        
        # 2. Draw the "Learned" Green Box
        # Since this is pixels, we estimate box size based on the cluster spread
        # Or you can hardcode a size like 100x50 pixels if you know the scale
        width = (np.max(xy[:, 0]) - np.min(xy[:, 0])) + 20 # Add padding
        height = (np.max(xy[:, 1]) - np.min(xy[:, 1])) + 20
        
        # Ensure minimum size (to avoid tiny boxes on partial detections)
        width = max(width, 40)
        height = max(height, 40)
        
        rect = patches.Rectangle(
            (cx - width/2, cy - height/2), width, height,
            linewidth=2, edgecolor='#00FF00', facecolor='#00FF0044'
        )
        ax.add_patch(rect)
        
        # Label it
        ax.text(cx, cy, f"P-{spot_id}", color='white', 
                ha='center', va='center', fontweight='bold', fontsize=8)
        
        spot_id += 1

    plt.tight_layout()
    plt.show()
    print("Graph generated. Save this image for your thesis!")

if __name__ == "__main__":
    process_video()