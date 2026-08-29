import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_mine_data(num_nodes=20, time_steps=5000, seed=42):
    np.random.seed(seed)
    
    # Define 4x5 spatial grid over mine panel (Latitude/Longitude offset around base point)
    base_lat, base_lon = 23.6345, 86.1523  # Jharia Coalfield coordinates as baseline
    nodes = []
    grid_rows, grid_cols = 4, 5
    
    for r in range(grid_rows):
        for c in range(grid_cols):
            node_idx = r * grid_cols + c + 1
            nodes.append({
                'node_id': f"NODE_{node_idx:02d}",
                'grid_x': c,
                'grid_y': r,
                'lat': base_lat + (r * 0.0008),
                'lon': base_lon + (c * 0.0008)
            })
            
    start_time = datetime.now() - timedelta(minutes=time_steps)
    data_list = []

    # Pre-allocate base baseline metrics for each node
    node_baselines = {
        n['node_id']: {
            'disp': np.random.uniform(0.1, 1.5),
            'tilt_x': np.random.uniform(-0.5, 0.5),
            'tilt_y': np.random.uniform(-0.5, 0.5),
            'vibe': np.random.uniform(0.01, 0.08)
        } for n in nodes
    }

    # Primary epicenter nodes for simulated subsidence event (Cluster: NODE_07, NODE_08, NODE_12, NODE_13)
    epicenter_nodes = ["NODE_07", "NODE_08", "NODE_12", "NODE_13"]
    secondary_nodes = ["NODE_02", "NODE_03", "NODE_06", "NODE_09", "NODE_11", "NODE_14", "NODE_17", "NODE_18"]

    for t in range(time_steps):
        current_time = start_time + timedelta(minutes=t)
        
        # Inject dynamic subsidence event between t=2000 and t=3800
        is_event_period = 2000 <= t <= 3800
        event_progress = (t - 2000) / 1800.0 if is_event_period else 0.0

        for n in nodes:
            nid = n['node_id']
            base = node_baselines[nid]

            # Environmental periodic noise (e.g., thermal expansion during day)
            temp = 28.0 + 6.0 * np.sin(2 * np.pi * t / 1440) + np.random.normal(0, 0.2)
            
            # Default normal fluctuations
            disp = base['disp'] + np.random.normal(0, 0.05)
            tilt_x = base['tilt_x'] + np.random.normal(0, 0.02)
            tilt_y = base['tilt_y'] + np.random.normal(0, 0.02)
            vibration = base['vibe'] + np.abs(np.random.normal(0, 0.01))
            crack_level = 0
            label = "NORMAL"

            if is_event_period:
                if nid in epicenter_nodes:
                    # Exponential/Non-linear deformation profile
                    severity = np.power(event_progress, 2.2)
                    disp += severity * 45.0 + np.random.normal(0, 0.3)
                    tilt_x += severity * 12.0 + np.random.normal(0, 0.1)
                    tilt_y += severity * 9.5 + np.random.normal(0, 0.1)
                    vibration += severity * 1.8 + np.random.normal(0, 0.05)
                    
                    if event_progress > 0.7:
                        crack_level = 3 if event_progress > 0.85 else 2
                        label = "CRITICAL"
                    elif event_progress > 0.4:
                        crack_level = 1
                        label = "WARNING"
                    elif event_progress > 0.15:
                        label = "WATCH"

                elif nid in secondary_nodes:
                    # Attenuated/Lagging deformation for perimeter nodes
                    severity = np.power(event_progress, 2.0) * 0.45
                    disp += severity * 20.0 + np.random.normal(0, 0.2)
                    tilt_x += severity * 5.0 + np.random.normal(0, 0.08)
                    tilt_y += severity * 4.0 + np.random.normal(0, 0.08)
                    vibration += severity * 0.8 + np.random.normal(0, 0.03)
                    
                    if event_progress > 0.75:
                        label = "WARNING"
                        crack_level = 1
                    elif event_progress > 0.3:
                        label = "WATCH"

            # Enforce non-negative physical bounds
            disp = max(0.0, float(disp))
            vibration = max(0.0, float(vibration))

            data_list.append({
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'node_id': nid,
                'latitude': n['lat'],
                'longitude': n['lon'],
                'tilt_x': round(tilt_x, 3),
                'tilt_y': round(tilt_y, 3),
                'vibration': round(vibration, 4),
                'displacement': round(disp, 3),
                'crack_level': int(crack_level),
                'temperature': round(temp, 2),
                'label': label
            })

    df = pd.DataFrame(data_list)

    # Calculate temporal derivative metrics (Velocity & Derivatives) per node
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['node_id', 'timestamp_dt']).reset_index(drop=True)

    df['previous_displacement'] = df.groupby('node_id')['displacement'].shift(1).fillna(df['displacement'])
    df['displacement_rate'] = (df['displacement'] - df['previous_displacement']).round(4)
    
    df['total_tilt'] = np.sqrt(df['tilt_x']**2 + df['tilt_y']**2).round(3)
    df['previous_tilt'] = df.groupby('node_id')['total_tilt'].shift(1).fillna(df['total_tilt'])
    df['tilt_rate'] = (df['total_tilt'] - df['previous_tilt']).round(4)

    df['previous_vibration'] = df.groupby('node_id')['vibration'].shift(1).fillna(df['vibration'])
    df['vibration_change'] = (df['vibration'] - df['previous_vibration']).round(4)

    # Drop intermediate sorting column
    df = df.drop(columns=['timestamp_dt'])
    
    # Save dataset
    output_path = "data/raw_subsidence_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully with {len(df)} records at '{output_path}'.")
    print(f"Label distribution:\n{df['label'].value_counts()}")

if __name__ == "__main__":
    generate_synthetic_mine_data()