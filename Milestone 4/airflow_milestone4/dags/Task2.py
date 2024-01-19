import os
import pandas as pd
import functions as fn

def extract_additional_resources_task(filename): 
    if not os.path.exists("./data/all_location_coordinates.csv"):
        df = pd.read_csv(filename)
        api_key = "AIzaSyBXV_Q4_CWvV7btH9drTwc3BYRoj2GwozQ"
        coordinates_df = fn.gather_and_save_unique_coordinates(df, api_key)
        
