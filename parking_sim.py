import traci
import time
import random
import os

# --- CONFIGURATION ---
VALID_SPOTS = [10 + (i * 8) for i in range(20)]  
ILLEGAL_POSITIONS = [50, 100, 150]
CSV_FILENAME = "simulated_parking_data.csv"

def setup_cctv_camera():
    """
    Configures the SUMO GUI to look like a fixed CCTV camera.
    """
    # 1. Switch to "Real World" mode (Asphalt + Car Textures)
    # This makes it look like a video feed, not a schematic.
    traci.gui.setSchema("View #0", "real world")
    
    # 2. Set the Camera Angle (Zoom and Position)
    # We position it in the middle of our parking strip (approx x=100)
    # Zoom level 600% gives a good close-up view.
    traci.gui.setZoom("View #0", 600)
    traci.gui.setOffset("View #0", 120, 0)
    
    print("CCTV Camera Initialized.")

def run():
    print("Starting SUMO...")
    # We use sumo-gui to visualize it
    traci.start(["sumo-gui", "-c", "parking.sumocfg"])
    
    # --- ACTIVATE CCTV VIEW ---
    # We must wait 1 step for the GUI to load before setting the camera
    traci.simulationStep() 
    setup_cctv_camera()

    step = 0
    occupied_spots = {} 
    
    # Initialize CSV
    with open(CSV_FILENAME, "w") as f:
        f.write("Time,ID,X,Y\n")

    try:
        while step < 5000:
            traci.simulationStep()
            
            # --- 1. ARRIVAL LOGIC ---
            if random.random() < 0.05:
                free_spots = [pos for pos in VALID_SPOTS if pos not in occupied_spots]
                if free_spots:
                    target_pos = random.choice(free_spots)
                    veh_id = f"car_{step}"
                    try:
                        traci.route.add(f"route_{step}", ["E1"])
                        traci.vehicle.add(veh_id, f"route_{step}")
                        traci.vehicle.setStop(veh_id, "E1", pos=target_pos, laneIndex=0, duration=2000)
                        # Remove explicit setColor to let "Real World" textures work
                        # traci.vehicle.setColor(veh_id, (0, 255, 0)) 
                        occupied_spots[target_pos] = veh_id
                    except traci.TraCIException:
                        pass

            # --- 2. DEPARTURE LOGIC ---
            for pos, veh_id in list(occupied_spots.items()):
                try:
                    if veh_id not in traci.vehicle.getIDList() or traci.vehicle.getSpeed(veh_id) > 1.0:
                        del occupied_spots[pos]
                except:
                    del occupied_spots[pos]

            # --- 3. ILLEGAL PARKING LOGIC ---
            if random.random() < 0.01:
                bad_id = f"truck_{step}"
                try:
                    traci.route.add(f"bad_route_{step}", ["E1"])
                    traci.vehicle.add(bad_id, f"bad_route_{step}")
                    bad_pos = random.choice(ILLEGAL_POSITIONS)
                    traci.vehicle.setStop(bad_id, "E1", pos=bad_pos, laneIndex=1, duration=100)
                    traci.vehicle.setColor(bad_id, (255, 0, 0)) # Keep Red for Illegal to make it obvious
                except:
                    pass

            # --- 4. DATA LOGGING ---
            if step % 60 == 0:
                current_data_batch = []
                for veh in traci.vehicle.getIDList():
                    if traci.vehicle.getSpeed(veh) < 0.1:
                        x, y = traci.vehicle.getPosition(veh)
                        current_data_batch.append(f"{step},{veh},{x},{y}\n")
                
                if current_data_batch:
                    with open(CSV_FILENAME, "a") as f:
                        f.writelines(current_data_batch)

            step += 1
            # Add a small delay so you can watch it like a real video
            time.sleep(0.05) 

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        try:
            traci.close()
        except:
            pass

if __name__ == "__main__":
    run()