# 874 Final Project

This project analyzes 911 emergency call data from the Seattle Police Department and explores potential correlations with historical weather data. Our goal is to uncover spatial, temporal, and environmental patterns using data mining techniques to improve insights into public safety and emergency response.

---

## Objectives

1. **Data Preprocessing**  
   Clean and filter the 911 dataset and merge it with weather data for the time range April 1, 2023 – April 1, 2025.

2. **Spatial Clustering**  
   Identify clusters in call locations based on time of day and geography.

3. **Association Analysis**  
   Discover patterns between emergency call types, weather conditions, and time.

4. **Predictive Modeling**  
   Develop models to predict emergency call volumes and locations based on time and weather conditions.

---

## 🗃️ Datasets

-   **Seattle Police 911 Call Data**  
    Source: [City of Seattle Open Data Portal](https://data.seattle.gov/Public-Safety/Call-Data/33kz-ixgy)

-   **Seattle Weather Data**  
    Source: [Meteostat API](https://dev.meteostat.net/)

---

## 🧪 Environment Setup

Install dependencies with [Mamba](https://mamba.readthedocs.io/en/latest/):

```bash
mamba env create -f environment.yml
mamba activate seattle911
```
