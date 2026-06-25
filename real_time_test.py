import serial
import joblib
import pandas as pd

print("======================================")
print("AI AIR QUALITY REAL-TIME SYSTEM")
print("======================================")

# Load models
env_model = joblib.load("models/environment_model.pkl")
aqi_model = joblib.load("models/aqi_model.pkl")
health_model = joblib.load("models/health_risk_model.pkl")

print("Models loaded successfully")

# Serial connection
try:
    ser = serial.Serial("COM8", 115200, timeout=1)
    ser.reset_input_buffer()
    print("Connected to ESP32 on COM8")
except Exception as e:
    print("Serial connection failed:", e)
    exit()

print("Waiting for sensor data...\n")


# -------------------------------
# Context-Aware Recommendation
# -------------------------------

def get_context_recommendation(aqi, environment):

    env = environment.lower()

    # AQI level classification
    if aqi <= 50:
        level = "good"
    elif aqi <= 100:
        level = "moderate"
    elif aqi <= 150:
        level = "unhealthy_sensitive"
    elif aqi <= 200:
        level = "unhealthy"
    elif aqi <= 300:
        level = "very_unhealthy"
    else:
        level = "hazardous"

    # Indoor environments
    if env in ["classroom", "office", "indoors_day", "indoors_night"]:
        if level in ["good", "moderate"]:
            return "Air is acceptable. Maintain ventilation."
        elif level == "unhealthy_sensitive":
            return "Open windows and improve ventilation."
        elif level == "unhealthy":
            return "Use air purifier and reduce occupancy."
        else:
            return "Stay indoors with air purification. Avoid activity."

    # Traffic environments
    elif env in ["traffic", "road", "highway"]:
        if level in ["good", "moderate"]:
            return "Air manageable. Avoid long exposure."
        elif level == "unhealthy_sensitive":
            return "Wear mask and limit exposure."
        elif level == "unhealthy":
            return "Avoid area if possible. Use mask."
        else:
            return "Highly polluted. Avoid completely."

    # Outdoor environments
    elif env in ["park", "outdoor", "playground"]:
        if level == "good":
            return "Safe for outdoor activities."
        elif level == "moderate":
            return "Light activity only."
        elif level == "unhealthy_sensitive":
            return "Sensitive people avoid outdoor activity."
        elif level == "unhealthy":
            return "Avoid outdoor exercise."
        else:
            return "Do not stay outside."

    # Default fallback
    else:
        if level == "good":
            return "Air quality is good."
        elif level == "moderate":
            return "Air quality acceptable."
        elif level == "unhealthy_sensitive":
            return "Take precautions."
        elif level == "unhealthy":
            return "Avoid exposure."
        else:
            return "Dangerous air conditions."


# -------------------------------
# Real-time loop
# -------------------------------

while True:

    try:
        line = ser.readline().decode(errors="ignore").strip()

        if not line:
            continue

        print("RAW:", line)

        # Ignore ESP32 boot messages
        if "load:" in line or "entry" in line:
            continue

        # Ignore header
        if "temp" in line.lower():
            continue

        values = line.split(",")

        if len(values) < 6:
            continue

        # Parse sensor values
        reading_id = int(values[0])
        temperature = float(values[1])
        pressure = float(values[2])
        gas = float(values[3])
        pm25 = float(values[4])
        esp_aqi = float(values[5])

        # AQI prediction
        aqi_features = pd.DataFrame(
            [[temperature, pressure, gas, pm25]],
            columns=["temperature", "pressure", "gas", "pm25"]
        )

        predicted_aqi = aqi_model.predict(aqi_features)[0]

        # Full feature set for other models
        full_features = pd.DataFrame(
            [[temperature, pressure, gas, pm25, predicted_aqi]],
            columns=["temperature", "pressure", "gas", "pm25", "aqi"]
        )

        # Predictions
        environment = env_model.predict(full_features)[0]
        health = health_model.predict(full_features)[0]

        # Context-aware recommendation
        recommendation = get_context_recommendation(predicted_aqi, environment)

        # ---------------- Output ----------------

        print("\n================================")
        print("Reading ID:", reading_id)

        print("\nSensor Data")
        print("--------------------------------")
        print("Temperature :", temperature, "C")
        print("Pressure    :", pressure, "hPa")
        print("Gas         :", gas)
        print("PM2.5       :", pm25)

        print("\nAI Predictions")
        print("--------------------------------")
        print("Environment   :", environment)
        print("Predicted AQI :", round(predicted_aqi, 2))
        print("ESP AQI       :", esp_aqi)
        print("Health Risk   :", health)

        print("\nSmart Recommendation")
        print("--------------------------------")
        print(recommendation)

        print("================================")

    except Exception as e:
        print("Error:", e)