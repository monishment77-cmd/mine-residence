
import serial
import json
import requests
import time

# Update 'COM3' to match your hardware USB port (e.g., 'COM4' or '/dev/ttyUSB0')
SERIAL_PORT = 'COM3'  
BAUD_RATE = 9600
API_URL = "http://127.0.0.1:8000/api/v1/telemetry/ingest"

def start_hardware_bridge():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"[HARDWARE BRIDGE] Connected to physical sensor node on {SERIAL_PORT}...")
        
        while True:
            if ser.in_waiting > 0:
                raw_line = ser.readline().decode('utf-8').strip()
                if raw_line.startswith("{") and raw_line.endswith("}"):
                    try:
                        telemetry_payload = json.loads(raw_line)
                        
                        # Post physical hardware data to FastAPI AI Backend
                        response = requests.post(API_URL, json=telemetry_payload)
                        res_data = response.json()
                        
                        prediction = res_data['risk_prediction']
                        print(f"[INGESTED] Node: {telemetry_payload['node_id']} | "
                              f"Disp: {telemetry_payload['displacement']}mm | "
                              f"AI Risk: {prediction['risk_score']}% | "
                              f"Status: {prediction['status']}")
                              
                    except json.JSONDecodeError:
                        print(f"[MALFORMED DATA] Received: {raw_line}")
            time.sleep(0.05)

    except serial.SerialException:
        print(f"[ERROR] Could not open {SERIAL_PORT}. Check USB cable connection and port name.")

if __name__ == "__main__":
    start_hardware_bridge()