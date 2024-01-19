import pandas as pd
import numpy as np
import os
from geopy.geocoders import GoogleV3
import time
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def read_dataSet(dataSet_path):
    return pd.read_csv(dataSet_path + "green_tripdata_2016-01.csv")

def update_lookup(mappings , lookup_table):
    for feature, mapping in mappings.items():
        lookup_table[feature] = mapping
    return lookup_table

def clean_column_names(df, rename_dict=None):
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    if rename_dict:
        rename_dict = {key.lower().replace(' ', '_'): value for key, value in rename_dict.items()}
        df = df.rename(columns=rename_dict)
    
    return df

def convert_to_datetime(df, columns):
    for column in columns:
        df[column] = pd.to_datetime(df[column])
    return df

def statistical_imputation(lookup_table, df, column, method='mean'):   
    if method == 'mean':
        imputed_value = df[column].mean()
    elif method == 'median':
        imputed_value = df[column].median()
    elif method == 'mode':
        imputed_value = df[column].mode()[0]  # mode() returns a Series, so we get the first entry
    else:
        raise ValueError("Method should be one of 'mean', 'median', or 'mode'.")
    
    df[column].fillna(imputed_value, inplace=True)
    lookup_table = update_lookup({column: {"null/nan": imputed_value}}, lookup_table)
    return df  

def fill_missing_with_zeros(lookup_table,df, feature):
    df[feature].fillna(0, inplace=True)
    lookup_table = update_lookup({feature: {"null/nan": 0}}, lookup_table)    
    return df, {feature: {"null/nan": 0}} , lookup_table 

def impute_missing_with_category(lookup_table ,df, feature, category):
    df[feature].fillna(category, inplace=True)
    lookup_table = update_lookup({feature: {pd.NA: category}}, lookup_table)

    return df, {feature: {pd.NA: category}}

def impute_specific_value(lookup_table, df, column, target_value, method='mean'):
    if method == 'mean':
        imputed_value = df.loc[df[column] != target_value, column].mean()
    elif method == 'median':
        imputed_value = df.loc[df[column] != target_value, column].median()
    elif method == 'mode':
        imputed_value = df.loc[df[column] != target_value, column].mode()[0]  # mode() returns a Series, so we get the first entry
    else:
        raise ValueError("Method should be one of 'mean', 'median', or 'mode'.")
    print(imputed_value)
    df.loc[df[column] == target_value, column] = imputed_value
    lookup_table= update_lookup({column: {target_value: imputed_value}}, lookup_table)

    return df, {column: {target_value: imputed_value}}


def adjust_column_for_multiple_values(df, col_to_adjust, adjustment_col, valid_values):
    # Convert the column to float for easy comparison
    df[col_to_adjust] = df[col_to_adjust].astype(float)

    # Create a mask for rows with invalid col_to_adjust values
    mask = ~df[col_to_adjust].isin(valid_values)

    # Adjust the adjustment_col based on the mask
    df.loc[mask, adjustment_col] -= df.loc[mask, col_to_adjust]

    # Set the invalid col_to_adjust values to 0
    df.loc[mask, col_to_adjust] = 0.0

    return df

def remove_rows_with_value(df, column_name, value_to_remove):
    return df[df[column_name] != value_to_remove]

def remove_rows_with_multiple_conditions(df, conditions):
    mask = pd.Series([True] * len(df))
    for feature, value in conditions:
        mask = mask & (df[feature] == value)
    return df[~mask]

def adjust_column_value_based_on_condition(df, condition_column, condition_value, target_column, desired_value):
    condition = (df[condition_column] == condition_value) & (df[target_column] != desired_value)
    df.loc[condition, target_column] = desired_value
    return df

def adjust_features_based_on_conditions(df, conditions, adjustments):
    mask = None
    for condition_column, operator, condition_value in conditions:
        condition_mask = df.eval(f"{condition_column} {operator} @condition_value", engine='python', local_dict={'condition_value': condition_value})
        mask = condition_mask if mask is None else mask & condition_mask

    if mask is not None:
        for feature, new_value in adjustments.items():
            df.loc[mask, feature] = new_value

    return df

def count_mismatch_rows(df, columns_to_sum, total_column, tolerance=1e-6):
    # Calculate the sum for each row
    row_sums = df[columns_to_sum].sum(axis=1)
    
    # Find rows where the sum isn't equal to the total
    mismatch_rows = df[abs(row_sums - df[total_column]) > tolerance]
    
    return len(mismatch_rows)

def remove_mismatched_rows(df, col_list, total_col, tolerance=1e-6):  
    # Calculate the row-wise sum of columns in col_list
    df['calculated_sum'] = df[col_list].sum(axis=1)
    
    # Determine rows where the difference between calculated_sum and total_col is greater than the tolerance
    mask = (df['calculated_sum'] - df[total_col]).abs() > tolerance
    
    # Remove those rows
    filtered_df = df[~mask].drop(columns=['calculated_sum'])
    
    return filtered_df

def replace_values(lookup_table ,df, feature, old_value, new_value):
    df_copy = df.copy()
    df_copy[feature] = df_copy[feature].replace(old_value, new_value)
    
    # Prepare the update details for the lookup table
    update_details = {feature: {old_value: new_value}}
    lookup_table= update_lookup(update_details, lookup_table)
    return df_copy

def filter_records_by_date_range(df, column_name, lower_bound=None, upper_bound=None):
    # Convert the bounds to datetime
    if lower_bound is not None:
        lower_bound = pd.to_datetime(lower_bound).normalize()
    if upper_bound is not None:
        upper_bound = pd.to_datetime(upper_bound).normalize() + pd.DateOffset(days=1) - pd.Timedelta(seconds=1)
    # Apply filtering
    if lower_bound:
        df = df[df[column_name] >= lower_bound]
    if upper_bound:
        df = df[df[column_name] <= upper_bound]
    return df

def remove_unwanted_values(df, feature, unwanted_value):
    return df[df[feature] != unwanted_value]

def fix_unwanted_values(df, feature, unwanted_value, replacement_value):
    # Create a copy of the DataFrame to avoid modifying the original DataFrame.
    fixed_df = df.copy()

    # Replace the unwanted value with the replacement value in the specified feature column.
    fixed_df.loc[fixed_df[feature] == unwanted_value, feature] = replacement_value

    # Return the DataFrame with the unwanted values fixed.
    return fixed_df

def impute_outliers_with_mean(df, feature, total_column=None, threshold=10):
    # Calculate the mean of positive values between 0 and threshold
    mean_val = df[(df[feature] > 0) & (df[feature] < threshold)][feature].mean()
    
    # Create a mask for rows that are outliers (values below -threshold or above threshold)
    outliers_mask = (df[feature] < -threshold) | (df[feature] > threshold)
    
    if total_column:
        # Adjust the total column based on the difference between the outlier and mean value
        df.loc[outliers_mask, total_column] = df.loc[outliers_mask, total_column] - df.loc[outliers_mask, feature] + mean_val
    
    # Impute the outliers with the mean value
    df.loc[outliers_mask, feature] = mean_val
    
    return df

def impute_outliers_with_mean_iqr(df, feature, total_column=None):    
    # Calculate the IQR and boundaries
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Create a mask for rows that are outliers
    outliers_mask = (df[feature] < lower_bound) | (df[feature] > upper_bound)
    
    # Calculate the mean of values within the IQR range
    mean_val = df[~outliers_mask][feature].mean()
    
    if total_column:
        # Adjust the total column based on the difference between the outlier and mean value
        df.loc[outliers_mask, total_column] = df.loc[outliers_mask, total_column] - df.loc[outliers_mask, feature] + mean_val
    
    # Impute the outliers with the mean value
    df.loc[outliers_mask, feature] = mean_val
    
    return df

import calendar

def add_weekly_columns(df, datetime_column):
    # Determine the start of the week for the earliest date in the dataframe
    first_date = df[datetime_column].min()
    start_of_first_week = first_date - pd.to_timedelta(first_date.dayofweek, unit='D')

    # Compute the day difference to the start of the first week
    day_difference = (df[datetime_column] - start_of_first_week).dt.days

    # Use this difference to determine the week number
    df['week_number'] = (day_difference // 7) + 1

    week_start = (df[datetime_column] - pd.to_timedelta(df[datetime_column].dt.dayofweek, unit='D')).dt.date
    week_end = (week_start + pd.DateOffset(days=6)).dt.date
    df['week_range'] = week_start.astype(str) + ' / ' + week_end.astype(str)
    
    return df

def equal_width_discretization(df, column_name, bins=10, new_column_name=None, labels=None): 
    # If no labels provided, use numerical bin labels
    if labels is None:
        labels = [i for i in range(1, bins + 1)]
    elif len(labels) != bins:
        raise ValueError(f"Number of labels ({len(labels)}) should match the number of bins ({bins}).")

    # If no new column name provided, overwrite the original column
    if new_column_name is None:
        new_column_name = column_name
    
    # Check if the column is of datetime type
    if df[column_name].dtype == 'datetime64[ns]':
        min_date = df[column_name].min()
        
        # Creating bins with 7 day width using pd.to_timedelta
        bin_edges = [min_date + pd.to_timedelta(7 * i, unit='D') for i in range(bins + 1)]
        series_to_cut = df[column_name]
    else:
        series_to_cut = df[column_name]
        bin_edges = bins
    
    # Use pandas cut function to create the discretized column
    df[new_column_name] = pd.cut(series_to_cut, bins=bin_edges, labels=labels, include_lowest=True)
    
    return df

def equal_width_discretization_range(df, column_name, bins=10, new_column_name=None):
    # If no new column name provided, overwrite the original column
    if new_column_name is None:
        new_column_name = column_name
    
    # Check if the column is of datetime type
    if df[column_name].dtype == 'datetime64[ns]':
        min_date = df[column_name].min()
        
        # Creating bins with 7 day width using pd.to_timedelta
        bin_edges = [min_date + pd.to_timedelta(7 * i, unit='D') for i in range(bins + 1)]
        series_to_cut = df[column_name]
    else:
        series_to_cut = df[column_name]
        bin_edges = bins
    
    # Generate labels for the date ranges
    labels = [f"{bin_edges[i].strftime('%Y-%m-%d')} / {(bin_edges[i+1] - pd.to_timedelta(1, unit='D')).strftime('%Y-%m-%d')}" for i in range(len(bin_edges)-1)]
    
    # Use pandas cut function to create the discretized column
    df[new_column_name] = pd.cut(series_to_cut, bins=bin_edges, labels=labels, include_lowest=True)
    
    return df

def get_coordinates(city_name, geolocator):
    location = geolocator.geocode(city_name)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None
    
def get_coordinates_google(location, api_key):
    geolocator = GoogleV3(api_key=api_key)

    # Split location into borough and zone
    parts = location.split(',')
    full_location = parts[0] + ", " + parts[1]
    
    try:
        # Try to geocode using full location (borough, zone)
        location_obj = geolocator.geocode(full_location, timeout=10)
        if location_obj:
            return location_obj.latitude, location_obj.longitude
        
        # If that fails, try just the borough
        location_obj = geolocator.geocode(parts[0], timeout=10)
        if location_obj:
            return location_obj.latitude, location_obj.longitude
    except Exception as e:
        print(f"Error fetching coordinates for {location}: {str(e)}")
        return None, None

    time.sleep(1)  # To prevent hitting request limits
    return None, None
    
def gather_and_save_unique_coordinates(df, api_key, pu_column='pickup_location', do_column='dropoff_location', filepath="./data/all_location_coordinates.csv"):
    """
    Gathers unique GPS coordinates for the given pickup and drop-off columns and saves them to a CSV.
    """
    
    # Extract unique values from both columns
    unique_pu_locations = df[pu_column].unique()
    unique_do_locations = df[do_column].unique()

    # Combine and deduplicate
    all_unique_locations = set(unique_pu_locations) | set(unique_do_locations)

    # If CSV doesn't exist, fetch coordinates and save to CSV
    if not os.path.exists(filepath):
        coordinates = {}
        for location in all_unique_locations:
            coords = get_coordinates_google(location, api_key)
            if coords:
                coordinates[location] = coords
        # Save to CSV
        coordinates_df = pd.DataFrame.from_dict(coordinates, orient='index', columns=['Latitude', 'Longitude'])
        coordinates_df.to_csv(filepath)
    else:
        coordinates_df = pd.read_csv(filepath, index_col=0)
    
    return coordinates_df

def populate_lat_long(df, prefix, coordinates_df):
    df[prefix + '_lat'] = df[prefix + '_location'].map(coordinates_df['Latitude'])
    df[prefix + '_long'] = df[prefix + '_location'].map(coordinates_df['Longitude'])
    return df

def encode_features(df, features, method):
    """
    Encode given columns of a dataframe.
    
    Parameters:
    - df: DataFrame to be encoded.
    - features: List of column names to be encoded.
    - method: Encoding method - either 'onehot' or 'label'.
    
    Returns:
    - DataFrame with encoded features.
    - Dictionary of mappings from original to encoded values for each feature.
    """
    df_copy = df.copy()
    mappings = {}

    # One-Hot Encoding
    if method == 'onehot':
        one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)
        
        df_encoded = pd.DataFrame(one_hot_encoder.fit_transform(df_copy[features]), 
                                  columns=one_hot_encoder.get_feature_names_out(features),
                                  index=df_copy.index)  # Important: Maintain the original index
        
        # Mapping for one-hot encoding - for each feature, a dict of category to encoded column
        for feature, cats in zip(features, one_hot_encoder.categories_):
            mappings[feature] = {cat: f"{feature}_{cat}" for cat in cats[1:]}  # Exclude first category
        
        df_copy = pd.concat([df_copy, df_encoded], axis=1)
        df_copy.drop(features, axis=1, inplace=True)  # Remove original columns after encoding
    
    # Label Encoding
    elif method == 'label':
        for feature in features:
            label_encoder = LabelEncoder()
            df_copy[feature] = label_encoder.fit_transform(df_copy[feature])  # In-place modification
            mappings[feature] = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    else:
        raise ValueError("Method must be either 'onehot' or 'label'")
    
    return df_copy, mappings

def add_features(df):
    
    # 1. Calculate Duration in hours
    df['duration_hours'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() / 3600
    
    # 2. Identify if the trip was on a weekend
    df['is_weekend'] = df['pickup_datetime'].dt.weekday >= 5  # 5 for Saturday, 6 for Sunday
    
    # 3. Calculate Average Speed in miles per hour
    df['avg_speed_mph'] = df['trip_distance'] / df['duration_hours']
    # Set avg_speed_mph to -1 where trip_distance is -1
    df['avg_speed_mph'] = np.where(df['trip_distance'] == -1, -1, df['avg_speed_mph'])
    
    return df

