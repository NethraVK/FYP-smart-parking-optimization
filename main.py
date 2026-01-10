import cv2
import numpy as np
import math
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
import glob

# --- NEW IMPORTS ---
# We now import your helper scripts from the 'src' folder
from src import parking
from src import parking_visualize

# --- CONFIGURATION UPDATE ---
# Point to the new location of the model
YOLO_MODEL = "models/yolo11n.pt"
# --- CONFIGURATION ---
# IMPORTANT: Put the path to your folder of images here
DATASET_PATH = "data/inputs/Parking_Lot_1/*.jpg" 


# --- TUNING PARAMETERS ---
CONFIDENCE_THRESHOLD = 0.4  # Slightly lower to catch cars in shadows
CLUSTERING_RADIUS = 15      # How close points must be to form a spot
MIN_SAMPLES_TO_LEARN = 10   # Minimum times a car must be seen to "map" a spot
MERGE_DISTANCE = 30         # Merge duplicate spots closer than this
OCCUPANCY_DISTANCE = 25     # If a car is this close to a spot center -> Occupied

def main():
    # Get list of images
    image_files = sorted(glob.glob(DATASET_PATH))
    if not image_files:
        print(f"Error: No images found at {DATASET_PATH}")
        return

    print("=== STEP 1: AUTO-LEARNING INFRASTRUCTURE ===")
    learned_spots = run_learning_phase(image_files)
    
    if not learned_spots:
        print("System failed to learn any spots. Try adjusting MIN_SAMPLES.")
        return

    print(f"\n=== STEP 2: STARTING LIVE MONITORING SYSTEM ===")
    print(f"Tracking {len(learned_spots)} Parking Spaces...")
    run_monitoring_phase(image_files, learned_spots)

def run_learning_phase(image_files):
    """
    Scans the image folder to map out where the parking spots are.
    """
    model = YOLO(YOLO_MODEL)
    long_term_memory = []
    
    total_images = len(image_files)
    print(f"Processing {total_images} images for calibration...")

    for i, img_path in enumerate(image_files):
        # Skip images to speed up learning (Process every 3rd image)
        if i % 3 != 0: continue
        
        frame = cv2.imread(img_path)
        if frame is None: continue

        # Progress bar
        if i % 50 == 0:
            print(f"Scanning... {int(i/total_images*100)}%")

        # Detect
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, classes=[2, 7], verbose=False)
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Size Filter (Ignore trucks/noise)
            w, h = x2 - x1, y2 - y1
            if w * h < 500 or w * h > 60000: continue
            
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            long_term_memory.append([cx, cy])

    # --- DATA ANALYSIS (DBSCAN + MERGE) ---
    print("Analyzing spatial patterns...")
    if not long_term_memory: return []
    
    data = np.array(long_term_memory)
    
    # 1. Cluster
    db = DBSCAN(eps=CLUSTERING_RADIUS, min_samples=MIN_SAMPLES_TO_LEARN).fit(data)
    labels = db.labels_
    
    # 2. Extract Centroids
    unique_labels = set(labels)
    initial_spots = []
    for k in unique_labels:
        if k == -1: continue
        
        class_member_mask = (labels == k)
        xy = data[class_member_mask]
        
        # Filter huge/tiny clusters (Traffic lanes)
        x_min, y_min = np.min(xy, axis=0)
        x_max, y_max = np.max(xy, axis=0)
        if (x_max - x_min) > 80 or (y_max - y_min) > 80: continue 
        
        cx = np.mean(xy[:, 0])
        cy = np.mean(xy[:, 1])
        initial_spots.append((cx, cy))
        
    # 3. Merge Duplicates
    final_spots = []
    for (new_x, new_y) in initial_spots:
        is_duplicate = False
        for i, (exist_x, exist_y) in enumerate(final_spots):
            dist = math.sqrt((new_x - exist_x)**2 + (new_y - exist_y)**2)
            if dist < MERGE_DISTANCE:
                final_spots[i] = ((exist_x + new_x)/2, (exist_y + new_y)/2)
                is_duplicate = True
                break
        if not is_duplicate:
            final_spots.append((new_x, new_y))
            
    print(f"SUCCESS: Map created with {len(final_spots)} spots.")
    return final_spots

def run_monitoring_phase(image_files, spots):
    """
    Loops through images again, detects cars, and updates the dashboard.
    """
    model = YOLO(YOLO_MODEL)
    
    cv2.namedWindow("Smart Parking System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Smart Parking System", 1280, 720) 

    for i, img_path in enumerate(image_files):
        frame = cv2.imread(img_path)
        if frame is None: continue

        # 1. Detect Current Cars
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, classes=[2, 7], verbose=False)
        
        current_cars = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            current_cars.append((cx, cy))

        # 2. Check Occupancy
        free_spots = 0
        occupied_spots = 0
        
        for idx, (sx, sy) in enumerate(spots):
            is_occupied = False
            color = (0, 255, 0) # Green (Free)
            
            # Check distance to ALL current cars
            for (cx, cy) in current_cars:
                dist = math.sqrt((sx - cx)**2 + (sy - cy)**2)
                if dist < OCCUPANCY_DISTANCE:
                    is_occupied = True
                    break
            
            if is_occupied:
                occupied_spots += 1
                color = (0, 0, 255) # Red (Occupied)
                cv2.circle(frame, (int(sx), int(sy)), 15, color, -1)
            else:
                free_spots += 1
                cv2.circle(frame, (int(sx), int(sy)), 15, color, 2)
                
            # Draw Spot ID
            cv2.putText(frame, str(idx), (int(sx)-5, int(sy)+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 3. Draw Dashboard
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (400, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        cv2.putText(frame, f"TOTAL SPOTS: {len(spots)}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"OCCUPIED: {occupied_spots}", (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"FREE: {free_spots}", (200, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Smart Parking System", frame)
        
        # Wait 100ms between frames to simulate video speed
        # Press 'q' to quit
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()