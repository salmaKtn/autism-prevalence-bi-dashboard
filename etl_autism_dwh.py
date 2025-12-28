# ETL Script - Autism Research Data Warehouse
# Author: Salma
# Date: December 2024

import pandas as pd 
import psycopg2
from psycopg2 import sql 
import re 
import numpy as np 

# Database configuration
DB_config = {
    'host': 'localhost',
    'database': 'DWH_ASD',
    'user': 'postgres',
    'password': 'admin'
}

Excel_file = 'AUTISM.xlsx'

# EXTRACT
print("-" * 60)
print("EXTRACT: Loading Excel File")
print("-" * 60)

df = pd.read_excel(Excel_file)
print(f"Loaded {len(df)} rows and {len(df.columns)} columns\n")

# TRANSFORM
print("-" * 60)
print("TRANSFORM: Building Dimension Tables")
print("-" * 60)

# DIM_Time
print("Building DIM_Time...")

def get_decade(year):
    decade_start = (year // 10) * 10
    return f"{decade_start}'s"

def get_period(year):
    if year < 2000:
        return "Pre-2000"
    elif year < 2010:
        return "2000-2010"
    else:
        return "2010-2020"

unique_years = df['Year Published'].dropna().unique()
dim_time = pd.DataFrame({'year': sorted(unique_years)})    
dim_time['decade'] = dim_time['year'].apply(get_decade)
dim_time['period'] = dim_time['year'].apply(get_period)
dim_time.insert(0, 'time_key', range(1, len(dim_time) + 1))

print(f"Created {len(dim_time)} time records")
print(f"Year range: {dim_time['year'].min()} - {dim_time['year'].max()}\n")

# DIM_Location
print("Building DIM_Location...")

df['Area(s)'] = df['Area(s)'].fillna('Unknown')

location_df = df[['Country', 'Area(s)']].copy()
location_df.columns = ['country', 'area']
location_df['is_nationwide'] = location_df['area'].str.lower().str.contains('nationwide|national', na=False)

dim_location = location_df.drop_duplicates().reset_index(drop=True)
dim_location.insert(0, 'location_key', range(1, len(dim_location) + 1))

print(f"Created {len(dim_location)} location records")
print(f"Countries: {dim_location['country'].nunique()}\n")

# DIM_Methodology
print("Building DIM_Methodology...")

methodology_df = df[['Case Identification Method', 'Case Criterion']].copy()
methodology_df.columns = ['case_identification_method', 'case_criterion']
methodology_df['case_identification_method'] = methodology_df['case_identification_method'].fillna('Not Specified')
methodology_df['case_criterion'] = methodology_df['case_criterion'].fillna('Not Specified')

dim_methodology = methodology_df.drop_duplicates().reset_index(drop=True)
dim_methodology.insert(0, 'methodology_key', range(1, len(dim_methodology) + 1))

print(f"Created {len(dim_methodology)} methodology records\n")

# DIM_Demographics
print("Building DIM_Demographics...")

def extract_age_range(age_text):
    if pd.isna(age_text):
        return None, None
    
    age_text = str(age_text).strip()
    
    match = re.search(r'(\d+\.?\d*)\s*(?:to|-)\s*(\d+\.?\d*)', age_text)
    if match:
        min_val = float(match.group(1))
        max_val = float(match.group(2))
        return int(round(min_val)), int(round(max_val))
    
    match = re.search(r'(\d+\.?\d*)', age_text)
    if match:
        age = int(round(float(match.group(1))))
        return age, age
    
    return None, None

age_data = df['Age Range'].apply(extract_age_range)
demographics_df = pd.DataFrame({
    'age_range': df['Age Range'].fillna('Not Specified'),
    'min_age': [x[0] for x in age_data],
    'max_age': [x[1] for x in age_data]
})

dim_demographics = demographics_df.drop_duplicates().reset_index(drop=True)
dim_demographics.insert(0, 'demographic_key', range(1, len(dim_demographics) + 1))

print(f"Created {len(dim_demographics)} demographic records\n")

# FACT Table
print("Building FACT_ASD_Studies...")

fact_table = df.copy()

fact_table = fact_table.merge(
    dim_time[['time_key', 'year']], 
    left_on='Year Published', 
    right_on='year', 
    how='left'
).drop('year', axis=1)

fact_table['Area(s)'] = fact_table['Area(s)'].fillna('Unknown')
fact_table = fact_table.merge(
    dim_location[['location_key', 'country', 'area']], 
    left_on=['Country', 'Area(s)'], 
    right_on=['country', 'area'], 
    how='left'
).drop(['country', 'area'], axis=1)

temp_method = fact_table[['Case Identification Method', 'Case Criterion']].fillna('Not Specified')
fact_table = fact_table.merge(
    dim_methodology[['methodology_key', 'case_identification_method', 'case_criterion']], 
    left_on=['Case Identification Method', 'Case Criterion'], 
    right_on=['case_identification_method', 'case_criterion'], 
    how='left'
).drop(['case_identification_method', 'case_criterion'], axis=1)

fact_table['Age Range'] = fact_table['Age Range'].fillna('Not Specified')
fact_table = fact_table.merge(
    dim_demographics[['demographic_key', 'age_range']], 
    left_on='Age Range', 
    right_on='age_range', 
    how='left'
).drop('age_range', axis=1)

fact_asd = pd.DataFrame({
    'study_id': range(1, len(fact_table) + 1),
    'time_key': fact_table['time_key'],
    'location_key': fact_table['location_key'],
    'methodology_key': fact_table['methodology_key'],
    'demographic_key': fact_table['demographic_key'],
    'prevalence_per_1000': fact_table['ASD Prevalence Estimate per 1,000'],
    'sample_size': fact_table['Sample Size'],
    'number_of_cases': fact_table['Number of Cases'],
    'male_female_ratio': fact_table['Male:Female Sex Ratio'],
    'author': fact_table['Author'],
    'title': fact_table['Title'],
    'confidence_interval': fact_table['Confidence Interval (CI)']
})

print(f"Created {len(fact_asd)} fact records\n")
print("Transformation complete\n")

# LOAD
print("=" * 60)
print("LOAD: Inserting Data into PostgreSQL")
print("=" * 60)

conn = psycopg2.connect(**DB_config)
cursor = conn.cursor()
print("Connected to database\n")

try:
    print("Clearing existing data...")
    cursor.execute("TRUNCATE TABLE FACT_ASD_Studies CASCADE;")
    cursor.execute("TRUNCATE TABLE DIM_Time CASCADE;")
    cursor.execute("TRUNCATE TABLE DIM_Location CASCADE;")
    cursor.execute("TRUNCATE TABLE DIM_Methodology CASCADE;")
    cursor.execute("TRUNCATE TABLE DIM_Demographics CASCADE;")
    conn.commit()
    
    print("Loading DIM_Time...")
    for _, row in dim_time.iterrows():
        cursor.execute(
            "INSERT INTO DIM_Time (year, decade, period) VALUES (%s, %s, %s)",
            (int(row['year']), row['decade'], row['period'])
        )
    conn.commit()
    print(f"Loaded {len(dim_time)} records")
    
    print("Loading DIM_Location...")
    for _, row in dim_location.iterrows():
        cursor.execute(
            "INSERT INTO DIM_Location (country, area, is_nationwide) VALUES (%s, %s, %s)",
            (row['country'], row['area'], bool(row['is_nationwide']))
        )
    conn.commit()
    print(f"Loaded {len(dim_location)} records")
    
    print("Loading DIM_Methodology...")
    for _, row in dim_methodology.iterrows():
        cursor.execute(
            "INSERT INTO DIM_Methodology (case_identification_method, case_criterion) VALUES (%s, %s)",
            (row['case_identification_method'], row['case_criterion'])
        )
    conn.commit()
    print(f"Loaded {len(dim_methodology)} records")
    
    print("Loading DIM_Demographics...")
    for _, row in dim_demographics.iterrows():
        min_age = int(row['min_age']) if pd.notna(row['min_age']) else None
        max_age = int(row['max_age']) if pd.notna(row['max_age']) else None
        cursor.execute(
            "INSERT INTO DIM_Demographics (age_range, min_age, max_age) VALUES (%s, %s, %s)",
            (row['age_range'], min_age, max_age)
        )
    conn.commit()
    print(f"Loaded {len(dim_demographics)} records")
    
    print("Loading FACT_ASD_Studies...")
    
    cursor.execute("SELECT time_key, year FROM DIM_Time")
    time_db = {year: key for key, year in cursor.fetchall()}
    
    cursor.execute("SELECT location_key, country, area FROM DIM_Location")
    location_db = {(country, area): key for key, country, area in cursor.fetchall()}
    
    cursor.execute("SELECT methodology_key, case_identification_method, case_criterion FROM DIM_Methodology")
    methodology_db = {(method, criterion): key for key, method, criterion in cursor.fetchall()}
    
    cursor.execute("SELECT demographic_key, age_range FROM DIM_Demographics")
    demographics_db = {age_range: key for key, age_range in cursor.fetchall()}
    
    for idx, row in df.iterrows():
        time_key = time_db.get(row['Year Published'])
        
        country_val = row['Country']
        area_val = row['Area(s)'] if pd.notna(row['Area(s)']) else 'Unknown'
        location_key = location_db.get((country_val, area_val))
        
        method_val = row['Case Identification Method'] if pd.notna(row['Case Identification Method']) else 'Not Specified'
        criterion_val = row['Case Criterion'] if pd.notna(row['Case Criterion']) else 'Not Specified'
        methodology_key = methodology_db.get((method_val, criterion_val))
        
        age_val = row['Age Range'] if pd.notna(row['Age Range']) else 'Not Specified'
        demographic_key = demographics_db.get(age_val)
        
        if None in [time_key, location_key, methodology_key, demographic_key]:
            continue
        
        sample_size = int(row['Sample Size']) if pd.notna(row['Sample Size']) else None
        num_cases = int(row['Number of Cases']) if pd.notna(row['Number of Cases']) else None
        ratio = float(row['Male:Female Sex Ratio']) if pd.notna(row['Male:Female Sex Ratio']) else None
        ci = row['Confidence Interval (CI)'] if pd.notna(row['Confidence Interval (CI)']) else None
        
        cursor.execute("""
            INSERT INTO FACT_ASD_Studies (
                time_key, location_key, methodology_key, demographic_key,
                prevalence_per_1000, sample_size, number_of_cases, male_female_ratio,
                author, title, confidence_interval
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            time_key, location_key, methodology_key, demographic_key,
            float(row['ASD Prevalence Estimate per 1,000']), 
            sample_size, num_cases, ratio,
            row['Author'], row['Title'], ci
        ))
    
    conn.commit()
    print(f"Loaded {len(df)} records")
    
    print("\nVerifying load...")
    cursor.execute("SELECT COUNT(*) FROM DIM_Time")
    print(f"DIM_Time: {cursor.fetchone()[0]} rows")
    
    cursor.execute("SELECT COUNT(*) FROM DIM_Location")
    print(f"DIM_Location: {cursor.fetchone()[0]} rows")
    
    cursor.execute("SELECT COUNT(*) FROM DIM_Methodology")
    print(f"DIM_Methodology: {cursor.fetchone()[0]} rows")
    
    cursor.execute("SELECT COUNT(*) FROM DIM_Demographics")
    print(f"DIM_Demographics: {cursor.fetchone()[0]} rows")
    
    cursor.execute("SELECT COUNT(*) FROM FACT_ASD_Studies")
    print(f"FACT_ASD_Studies: {cursor.fetchone()[0]} rows")
    
    print("\n" + "=" * 60)
    print("ETL Process Complete")
    print("=" * 60)
    
except Exception as e:
    print(f"Error: {e}")
    conn.rollback()
    
finally:
    cursor.close()
    conn.close()
