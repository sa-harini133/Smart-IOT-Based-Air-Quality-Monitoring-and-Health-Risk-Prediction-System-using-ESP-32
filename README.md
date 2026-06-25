
# Smart IoT-Based Air Quality Monitoring & Health Risk Prediction System

## Overview

An intelligent IoT-based air quality monitoring system built using the **ESP32** and a multi-sensor array to collect real-time environmental data, predict Air Quality Index (AQI), classify health risks, identify the surrounding environment, and forecast air quality one hour ahead using machine learning models.

Unlike traditional AQI monitors that only display sensor readings, this system combines embedded sensing with predictive analytics to generate **context-aware health recommendations** for users in real time.

---

## Features

* Real-time air quality monitoring using ESP32
* Multi-sensor environmental data acquisition
* AQI prediction using Machine Learning
* Health risk classification (Low, Moderate, High, Hazardous)
* Environment identification
* 1-hour AQI forecasting
* Context-aware health recommendations
* Wi-Fi-enabled IoT platform
* Low-cost and scalable embedded system

---

## System Architecture

```text
                   ESP32
                     │
     ┌───────────────┼───────────────┐
     │               │               │
 BMP280          MQ Gas Sensor     PM2.5 Sensor
     │               │               │
     └───────────────┼───────────────┘
                     │
             Sensor Data Collection
                     │
                     ▼
           Machine Learning Pipeline
       ┌─────────┬──────────┬──────────┐
       │         │          │
 AQI Prediction  Health Risk  Environment Detection
       └─────────┴──────────┴──────────┘
                     │
              AQI Forecasting
                     │
                     ▼
      Context-Aware Recommendation Engine
```

---

## Hardware Components

* ESP32 Development Board
* BMP280 Temperature & Pressure Sensor
* MQ Gas Sensor
* PM2.5 Dust Sensor
* Breadboard
* Jumper Wires
* USB Power Supply

---

## Machine Learning Pipeline

The system integrates three specialized machine learning models:

### AQI Prediction

Predicts a numerical Air Quality Index from real-time sensor readings.

### Health Risk Classification

Classifies current environmental conditions into:

* Low
* Moderate
* High
* Hazardous

### Environment Identification

Recognizes the deployment environment, including:

* Kitchen
* Classroom
* Construction Site
* Park / Outdoor
* Indoor AC Room
* Other indoor environments

---

## Intelligent Recommendation Engine

The outputs of all ML models are combined to generate environment-specific recommendations, including:

* Kitchen ventilation alerts
* Classroom ventilation suggestions
* Construction site mask recommendations
* Indoor air purifier activation
* AQI trend-based preventive health advice

---

## AQI Forecasting

In addition to monitoring current conditions, the system predicts air quality one hour ahead, enabling:

* Proactive health alerts
* Smarter scheduling of outdoor activities
* Trend-aware recommendations
* Early warning before hazardous conditions develop

---

## Technologies Used

### Embedded Systems

* ESP32
* Arduino IDE

### Sensors

* BMP280
* MQ Gas Sensor
* PM2.5 Sensor

### Machine Learning

* Python
* Scikit-learn
* Pandas
* NumPy

### Data Visualization

* Matplotlib

---




## Applications

* Smart Cities
* Indoor Air Quality Monitoring
* Public Health Surveillance
* Smart Buildings
* Industrial Safety
* Construction Site Monitoring
* Educational Institutions
* Healthcare Facilities

---

## Future Improvements

* Cloud dashboard for remote monitoring
* Mobile application
* Edge deployment of ML models on ESP32
* GPS-enabled pollution mapping
* Additional environmental sensors
* Long-term air quality analytics
* MQTT/ThingSpeak integration

---

## Results

* Successfully collected real-world environmental data using an ESP32-based sensor array.
* Trained three machine learning models for AQI prediction, health risk classification, and environment identification.
* Generated context-aware recommendations based on live sensor readings and predicted air quality trends.
* Demonstrated one-hour AQI forecasting for proactive health monitoring.

---

## Demo

A prototype demonstration video is included in this repository to showcase real-time sensor monitoring and intelligent air quality prediction.

---


