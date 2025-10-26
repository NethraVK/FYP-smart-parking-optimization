# FYP-smart-parking-optimization
ParkOptimize Dubai: A scalable AI system for real-time parking occupancy detection from CCTV streams. This project uses YOLOv10 to detect and track vehicles, providing instant occupancy data via a FastAPI WebSocket. Built for RTA-level scale with analytics powered by TimescaleDB.
## 1. The Problem: Parking Inefficiency in Dubai

Finding available parking in Dubai's high-density areas is a significant contributor to traffic congestion, wasted fuel, and driver frustration. While the RTA manages parking payments, this transaction data doesn't reflect the real-time physical occupancy of individual parking spots. This leads to several critical blind spots:

* **Spot Ambiguity:** Payment data confirms payment for a zone (e.g., 317A) but doesn't specify which of the potentially dozens of spots within that zone is actually occupied.
* **False Occupancy:** A driver might pay for 3 hours but leave after 30 minutes. Payment data incorrectly flags the spot as occupied for the full duration, hiding its availability.
* **"Illegal Parker" Problem:** A vehicle might occupy a spot without paying. Payment data incorrectly flags the spot as vacant, misleading drivers.

Standard computer vision projects attempting to solve this often fail at scale due to simplistic logic, lack of real-time updates, inability to store historical data, and poor performance on multiple video streams.

## 2. This Project's Solution:

This project builds a robust, scalable, and analytical AI system designed for city-wide deployment, providing the ground truth of parking occupancy. It leverages cutting-edge technology to overcome the failures of standard approaches:

* **Smarter Logic (Eliminating False Positives):**
    * **Standard:** Simple detection marks spots occupied even by cars just driving through.
    * **This Project:** Uses **YOLOv10 + ByteTrack** for object tracking. A spot is confirmed `OCCUPIED` only if a vehicle is tracked as stationary within its boundaries for a defined period (e.g., >10 seconds), drastically reducing false alerts.

* **Instant Real-Time Updates (Efficient & Scalable):**
    * **Standard:** Inefficient HTTP "polling" creates lag and network overhead.
    * **This Project:** Employs a **FastAPI WebSocket**. The server instantly *pushes* status changes (`VACANT`/`OCCUPIED`) to connected applications (web/mobile) the moment they occur, ensuring true real-time data with minimal resources.

* **Historical Analytics (Data-Driven Insights):**
    * **Standard:** No database; offers zero historical insight.
    * **This Project:** Logs every occupancy change event with a timestamp to **TimescaleDB**. This enables powerful analytics via dashboards to answer critical questions about peak times, spot utilization, average parking duration, and compliance rates.

* **Scalable by Design (Ready for City-Wide Deployment):**
    * **Standard:** Basic Python scripts choke on even a single video feed.
    * **This Project:** The core architecture is built for performance and designed for migration to **NVIDIA DeepStream**. This industry-standard SDK allows processing thousands of camera streams in parallel on optimized hardware, making a city-scale deployment feasible.

By providing accurate, real-time, and historical occupancy data, this project aims to be a foundational component for an intelligent urban mobility and parking optimization in Dubai.
