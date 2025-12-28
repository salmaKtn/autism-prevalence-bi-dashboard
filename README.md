# autism-prevalence-bi-dashboard
Autism Prevalence Research: Business Intelligence Analysis
A comprehensive BI project analyzing 142 peer-reviewed autism prevalence studies from the CDC database (1966-2020) across 45+ countries.
🎯 Project Overview
This project examines six decades of autism research to understand how prevalence rates have evolved and what factors influence reported findings. The analysis separates clinical patterns from research methodology to provide actionable insights for healthcare professionals and researchers.
🏗️ Architecture

Data Warehouse: PostgreSQL star schema with fact table (autism studies) and 4 dimension tables (temporal, geographic, methodological, demographic)
ETL Pipeline: Python-based data integration using PygramETL
OLAP Analysis: Multidimensional analysis with drill-down, roll-up, and slice operations
Dashboards: Interactive Atoti dashboards with dual-view design
Machine Learning: Random Forest regression for prevalence prediction and feature importance analysis

📊 Key Findings

Male-to-female diagnosis ratio: 400:1
Research activity peaked in 2010s (75 contributors) with sharp decline in 2020s
Diagnostic methodology is the strongest predictor of reported prevalence (28% feature importance)
242,118 total cases analyzed across studies

🛠️ Tech Stack

Database: PostgreSQL
ETL: Python (pandas, PygramETL)
Visualization: Atoti
ML: scikit-learn (Random Forest)
OLAP: Python (cube/atoti)


📈 Data Source
CDC Autism Prevalence Studies Database: https://data.cdc.gov/Public-Health-Surveillance/autism-prevalence-studies/9mw4-6adp
