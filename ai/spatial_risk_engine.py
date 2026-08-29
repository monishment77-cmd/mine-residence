import numpy as np
import pandas as pd
import joblib

class MineSpatialRiskEngine:
    def __init__(self, model_dir="models/"):
        # Load trained ML models and artifacts
        self.rf_model = joblib.load(f"{model_dir}risk_classifier.pkl")
        self.iso_forest = joblib.load(f"{model_dir}anomaly_detector.pkl")
        self.scaler = joblib.load(f"{model_dir}feature_scaler.pkl")
        self.feature_cols = joblib.load(f"{model_dir}feature_columns.pkl")
        
        # Spatial Grid Mapping (4x5 Grid = 20 Nodes)
        self.grid_rows = 4
        self.grid_cols = 5
        self.node_positions = {
            f"NODE_{r * self.grid_cols + c + 1:02d}": (r, c)
            for r in range(self.grid_rows) for c in range(self.grid_cols)
        }

    def predict_node_risk(self, node_data_dict):
        """Processes a single node payload and returns ML predictions."""
        df_input = pd.DataFrame([node_data_dict])
        
        # Compute dynamic features if missing
        if 'total_tilt' not in df_input.columns:
            df_input['total_tilt'] = np.sqrt(df_input['tilt_x']**2 + df_input['tilt_y']**2).round(3)
        
        X_features = df_input[self.feature_cols]
        X_scaled = self.scaler.transform(X_features)

        # 1. ML Anomaly Detection (-1 = Anomaly, 1 = Normal)
        is_anomaly = bool(self.iso_forest.predict(X_scaled)[0] == -1)

        # 2. ML Risk Probabilities & Classification
        probs = self.rf_model.predict_proba(X_features)[0]
        classes = list(self.rf_model.classes_)
        # Cast numpy types to native Python types so FastAPI/JSON can serialize them
        prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probs)}
        
        # Weighted Risk Score (0 - 100)
        risk_score = (
            prob_dict.get('NORMAL', 0) * 15 +
            prob_dict.get('WATCH', 0) * 45 +
            prob_dict.get('WARNING', 0) * 75 +
            prob_dict.get('CRITICAL', 0) * 95
        )
        
        status = str(self.rf_model.predict(X_features)[0])

        return {
            "node_id": node_data_dict['node_id'],
            "risk_score": round(float(risk_score), 2),
            "status": status,
            "is_anomaly": is_anomaly,
            "probabilities": prob_dict
        }

    def get_neighbor_nodes(self, node_id):
        """Finds immediate spatial neighbors on the mesh grid (Up, Down, Left, Right)."""
        if node_id not in self.node_positions:
            return []
        
        r, c = self.node_positions[node_id]
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols:
                n_id = f"NODE_{nr * self.grid_cols + nc + 1:02d}"
                neighbors.append(n_id)
        return neighbors

    def evaluate_mesh_network(self, all_nodes_data):
        """
        Analyzes the full surface mesh network to identify correlated high-risk zones.
        Distinguishes single sensor hardware failures from true spatial subsidence events.
        """
        node_results = {}
        for node in all_nodes_data:
            res = self.predict_node_risk(node)
            node_results[node['node_id']] = res

        high_risk_clusters = []
        visited = set()

        for node_id, res in node_results.items():
            if res['risk_score'] >= 60 and node_id not in visited:  # WARNING or CRITICAL threshold
                cluster = []
                queue = [node_id]
                visited.add(node_id)

                while queue:
                    curr = queue.pop(0)
                    cluster.append(curr)

                    for neighbor in self.get_neighbor_nodes(curr):
                        if neighbor in node_results and neighbor not in visited:
                            if node_results[neighbor]['risk_score'] >= 50: # Neighbor also showing strain
                                visited.add(neighbor)
                                queue.append(neighbor)

                if len(cluster) >= 2:  # 2 or more neighboring nodes confirming movement
                    high_risk_clusters.append({
                        "zone_name": f"Subsidence Zone {chr(65 + len(high_risk_clusters))}",
                        "affected_nodes": cluster,
                        "cluster_size": len(cluster),
                        "max_risk_score": max(node_results[n]['risk_score'] for n in cluster)
                    })

        # Calculate overall mine panel risk index
        avg_panel_risk = np.mean([res['risk_score'] for res in node_results.values()])
        max_panel_risk = np.max([res['risk_score'] for res in node_results.values()])

        return {
            "overall_mine_risk": round(float((avg_panel_risk * 0.4) + (max_panel_risk * 0.6)), 2),
            "node_predictions": node_results,
            "active_subsidence_zones": high_risk_clusters
        }

# Quick Self-Test
if __name__ == "__main__":
    engine = MineSpatialRiskEngine()
    
    # Test Payload simulating high risk on NODE_07 and neighboring NODE_08
    test_mesh = [
        {"node_id": "NODE_07", "tilt_x": 8.5, "tilt_y": 6.2, "vibration": 1.5, "displacement": 42.0, "crack_level": 2, "temperature": 31.0, "displacement_rate": 3.5, "tilt_rate": 0.8, "vibration_change": 0.4},
        {"node_id": "NODE_08", "tilt_x": 7.1, "tilt_y": 5.0, "vibration": 1.2, "displacement": 35.0, "crack_level": 1, "temperature": 30.5, "displacement_rate": 2.8, "tilt_rate": 0.6, "vibration_change": 0.3},
        {"node_id": "NODE_01", "tilt_x": 0.1, "tilt_y": 0.2, "vibration": 0.02, "displacement": 0.5, "crack_level": 0, "temperature": 28.0, "displacement_rate": 0.0, "tilt_rate": 0.0, "vibration_change": 0.0}
    ]
    
    output = engine.evaluate_mesh_network(test_mesh)
    print("Risk Engine Output:")
    print(f"Overall Mine Risk: {output['overall_mine_risk']}")
    print(f"Active Subsidence Zones Detected: {output['active_subsidence_zones']}")