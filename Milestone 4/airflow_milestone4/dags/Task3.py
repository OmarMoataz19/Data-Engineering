import os
import pandas as pd
import functions as fn
from sqlalchemy import create_engine

def integrate_load_to_postgres_csv_task (gps_file_path, afterEncoding_file_path, lookup_file_path):
    if not os.path.exists("./data/green_tripdata_2016-01_clean.csv"):
        coordinates_df = pd.read_csv(gps_file_path)

        df_copy = pd.read_csv(afterEncoding_file_path) 
        df_copy = fn.populate_lat_long(df_copy, 'pickup', coordinates_df)
        df_copy = fn.populate_lat_long(df_copy, 'dropoff', coordinates_df)

        df_copy.to_csv('./data/green_tripdata_2016-01_clean.csv',index=False) # integeration done
        lookUp = pd.read_csv(lookup_file_path)

        engine = create_engine('postgresql://root:root@pgdatabase:5432/green_taxisM4')
        if(engine.connect()):
            print('connected succesfully')
        else:
            print('failed to connect')
        try:
            df_copy.to_sql(name = 'green_taxi_01_2016_clean',con = engine,if_exists='fail')
            lookUp.to_sql(name = 'lookup_green_taxi_01_2016',con = engine,if_exists='fail')
        except ValueError as vx:
            print("Database tables already exists")



