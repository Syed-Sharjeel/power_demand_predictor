# AI Powered Load Forecasting and Report Generator
## Executive Summary
This AI-powered application predicts future electricity load consumption based on multiple factors such as weather, social trends, and historical usage data. The main objective is to support power utility departments by providing next-day load forecasts, enabling them to better prepare and adjust their systems accordingly. The generated reports help decision-makers plan for peak demand periods and optimize power distribution proactively.
## Introduction
The objective of the application is to aware the power utility department about the estimated load required for the next day. Power generation process is the initial and fundamental process performed by power utility departments. The problems of power disruption or arc flash in any area is caused by power shortage or excess respectively. To overcome the issue this application would be really helpful.
## Development Workflow
1.	**Data Ingestion** Gathering historical power consumption and weather data.
2.	**Data Cleaning & Feature Engineering** Apply preprocessing on the dataset to ensure the clean data, and apply feature engineering for better results.
3.	**Model Training** Used XGBoost model which is well-known for the algorithm in regression tasks.
4.	**Deployment & UI** Used Streamlit interface for UI and Streamlit Cloud for deployment for better user experience.
### Tech Stacks
1.	**Streamlit** Frontend User-Interface for better User-Experience.
2.	**Open-Meteo** API for real-time weather forecasting.
3.	**Nominatim** API for real-time Latitude, and Longitude coordinates of cities.
4.	**Pandas** Used for Data Manipulation and Cleaning.
5.	**Scikit-Learn** Used for XGBoost model and evaluation metrics.
## Future Enhancements
1.	**Optional Cloud Hosting Solution** Deploy on more advanced cloud-services or self-cloud services.
2.	**Enhanced Data Encryption** Incorporate more weather and other parameters which contributes to more accuracy.
3.	**Desktop & Mobile Application** Develop mobile friendly version and desktop application for on-the-go access.

## Acknowledgement
*This project is made for Hackathon organized by **Pak Angles** in Collaboration with **Pakistan Engineering Coucil (PEC)**, **iCodeGuru**, and **Aspire Pakistan**.* 
