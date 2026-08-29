import time
import random
import math
from datetime import datetime, timezone

class MineSensorSimulator:
    def __init__(self, num_nodes=20):
        self.num_nodes = num_nodes
        self.base_lat, self.base_lon = 23.6345, 86.1523
        self.grid_rows, self.grid_cols = 4, 5
        self.active_scenario = "NORMAL"
        self.epicenter = []
        self.nodes = {}
        self.reset_to_baseline()

    def reset_to_baseline(self):
        """Resets all nodes to clean baseline values."""
        self.nodes = {}
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                node_idx = r * self.grid_cols + c + 1
                node_id = f"NODE_{node_idx:02d}"
                self.nodes[node_id] = {
                    'node_id': node_id,
                    'lat': self.base_lat + (r * 0.0008),
                    'lon': self.base_lon + (c * 0.0008),
                    'displacement': random.uniform(0.1, 0.5),
                    'tilt_x': random.uniform(-0.05, 0.05),
                    'tilt_y': random.uniform(-0.05, 0.05),
                    'vibration': random.uniform(0.01, 0.03),
                    'crack_level': 0,
                    'temperature': 28.0,
                    'prev_disp': 0.2,
                    'prev_tilt': 0.05,
                    'prev_vibe': 0.02
                }
        self.epicenter = []

    def get_sim_neighbors(self, node_id: str) -> list:
        """Finds adjacent mesh nodes on the 4x5 grid."""
        node_num = int(node_id.split("_")[1]) - 1
        r, c = divmod(node_num, self.grid_cols)
        
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols:
                n_idx = nr * self.grid_cols + nc + 1
                neighbors.append(f"NODE_{n_idx:02d}")
        return neighbors

    def set_scenario(self, scenario_name: str):
        """Changes scenario and wipes previous strain values across all nodes."""
        valid_scenarios = ["NORMAL", "INCREASING_SUBSIDENCE", "CRACK_EVENT", "VIBRATION_EVENT", "CRITICAL_SUBSIDENCE"]
        if scenario_name in valid_scenarios:
            self.active_scenario = scenario_name
            self.reset_to_baseline()
            
            if scenario_name != "NORMAL":
                all_node_ids = list(self.nodes.keys())
                anchor_id = random.choice(all_node_ids)
                neighbors = self.get_sim_neighbors(anchor_id)
                self.epicenter = [anchor_id] + neighbors[:3]
                
            print(f"[SIMULATOR] Scenario: {scenario_name} | Epicenter: {self.epicenter}")

    def generate_telemetry_frame(self) -> list:
        """Generates continuous grid telemetry frame."""
        frame = []
        now_utc = datetime.now(timezone.utc).isoformat()
        
        for node_id, state in self.nodes.items():
            state['prev_disp'] = state['displacement']
            state['prev_tilt'] = math.sqrt(state['tilt_x']**2 + state['tilt_y']**2)
            state['prev_vibe'] = state['vibration']
            state['temperature'] = round(28.0 + 4.0 * math.sin(time.time() / 10.0) + random.normalvariate(0, 0.1), 2)

            if self.active_scenario == "NORMAL":
                state['displacement'] = max(0.1, state['displacement'] + random.normalvariate(0, 0.02))
                state['tilt_x'] += random.normalvariate(0, 0.01)
                state['tilt_y'] += random.normalvariate(0, 0.01)
                state['vibration'] = max(0.01, random.normalvariate(0.03, 0.005))
                state['crack_level'] = 0

            elif self.active_scenario == "INCREASING_SUBSIDENCE":
                if node_id in self.epicenter:
                    state['displacement'] = min(80.0, state['displacement'] + random.uniform(0.8, 2.5))
                    state['tilt_x'] += random.uniform(0.3, 0.8)
                    state['tilt_y'] += random.uniform(0.2, 0.6)
                    state['vibration'] = max(0.05, state['vibration'] + random.uniform(0.02, 0.08))
                    state['crack_level'] = 1 if state['displacement'] > 15.0 else 0
                else:
                    state['displacement'] += random.uniform(0.05, 0.2)

            elif self.active_scenario == "CRACK_EVENT":
                if node_id in self.epicenter:
                    state['crack_level'] = min(3, state['crack_level'] + 1)
                    state['tilt_x'] += random.uniform(0.5, 1.2)
                    state['vibration'] = random.uniform(0.3, 0.7)

            elif self.active_scenario == "VIBRATION_EVENT":
                state['vibration'] = random.uniform(1.2, 3.5)

            elif self.active_scenario == "CRITICAL_SUBSIDENCE":
                if node_id in self.epicenter:
                    state['displacement'] = min(150.0, state['displacement'] + random.uniform(4.0, 8.0))
                    state['tilt_x'] += random.uniform(1.5, 3.0)
                    state['tilt_y'] += random.uniform(1.2, 2.5)
                    state['vibration'] = random.uniform(1.5, 4.0)
                    state['crack_level'] = 3

            curr_tilt = round(math.sqrt(state['tilt_x']**2 + state['tilt_y']**2), 3)
            disp_rate = round(state['displacement'] - state['prev_disp'], 4)
            tilt_rate = round(curr_tilt - state['prev_tilt'], 4)
            vibe_change = round(state['vibration'] - state['prev_vibe'], 4)

            frame.append({
                "timestamp": now_utc,
                "node_id": node_id,
                "latitude": state['lat'],
                "longitude": state['lon'],
                "tilt_x": round(state['tilt_x'], 3),
                "tilt_y": round(state['tilt_y'], 3),
                "total_tilt": curr_tilt,
                "vibration": round(state['vibration'], 4),
                "displacement": round(state['displacement'], 3),
                "crack_level": int(state['crack_level']),
                "temperature": state['temperature'],
                "previous_displacement": round(state['prev_disp'], 3),
                "displacement_rate": disp_rate,
                "tilt_rate": tilt_rate,
                "vibration_change": vibe_change
            })

        return frame