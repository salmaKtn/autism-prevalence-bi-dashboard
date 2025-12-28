# Atoti Dashboard - Autism Research Data Warehouse
# Author: Salma
# Date: December 2024

import atoti as tt
import pandas as pd
from sqlalchemy import create_engine

DB_CONFIG = {
    'host': 'localhost',
    'database': 'DWH_ASD',
    'user': 'postgres',
    'password': 'admin'
}

connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"

print("=" * 80)
print("ATOTI DASHBOARD - AUTISM RESEARCH DATA WAREHOUSE")
print("=" * 80)

with tt.Session.start() as session:
    print("\nCreating Atoti session...")
    print("Session created")
    
    print("\nLoading data from PostgreSQL...")
    
    sql_query = """
    SELECT 
        f.study_id,
        f.prevalence_per_1000,
        COALESCE(f.sample_size, 0) as sample_size,
        COALESCE(f.number_of_cases, 0) as number_of_cases,
        COALESCE(f.male_female_ratio, 0) as male_female_ratio,
        f.author,
        f.title,
        t.year,
        t.decade,
        t.period,
        l.country,
        l.area,
        m.case_identification_method,
        m.case_criterion,
        d.age_range
    FROM FACT_ASD_Studies f
    JOIN DIM_Time t ON f.time_key = t.time_key
    JOIN DIM_Location l ON f.location_key = l.location_key
    JOIN DIM_Methodology m ON f.methodology_key = m.methodology_key
    JOIN DIM_Demographics d ON f.demographic_key = d.demographic_key
    """
    
    engine = create_engine(connection_string)
    df = pd.read_sql(sql_query, engine)
    engine.dispose()
    
    df = df.fillna({
        'sample_size': 0.0,
        'number_of_cases': 0.0,
        'male_female_ratio': 0.0
    })
    
    df['year'] = df['year'].astype(str)
    
    print(f"Loaded {len(df)} studies from database")
    
    print("\nCreating OLAP cube...")
    
    studies_table = session.read_pandas(
        df,
        table_name="autism_studies",
        keys=["study_id"]
    )
    
    cube = session.create_cube(studies_table)
    h = cube.hierarchies
    
    h["Country"] = [studies_table["country"]]
    h["Year"] = [studies_table["year"]]
    h["Decade"] = [studies_table["decade"]]
    h["Methodology"] = [studies_table["case_criterion"]]
    
    print("Cube created with hierarchies")
    
    print("\nCreating sample analyses...")
    
    m = cube.measures
    
    print("\nQuery 1: Average Prevalence by Country (Top 10)")
    query1 = cube.query(
        m["prevalence_per_1000.MEAN"],
        levels=[h["Country"]["country"]]
    )
    print(query1.head(10))
    
    print("\nQuery 2: Studies by Decade")
    query2 = cube.query(
        m["prevalence_per_1000.MEAN"],
        levels=[h["Decade"]["decade"]]
    )
    print(query2)
    
    print("\n" + "=" * 80)
    print("LAUNCHING INTERACTIVE DASHBOARD")
    print("=" * 80)
    print("\nKeep this window open while using the dashboard.")
    print("=" * 80)
    
    print(f"\nDashboard URL: {session.url}")
    print("\nOpen this URL in your browser to access the dashboard")
    print("Press Ctrl+C to stop the server")
    
    try:
        session.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")

print("Dashboard closed")
