import pandas as pd
import functions as fn
import os
import ingest as ing

# check if the clean dataset exists, if so terminate
# if not, continue with the cleaning process
filepath = "./data/green_tripdata_2016-01_clean.csv"
if os.path.exists(filepath):
    print("Cleaned Dataset already exists")
    ing.connectDB("./data/green_tripdata_2016-01_clean.csv", "./data/lookup_table_green_taxis.csv","green_taxi_01_2016", "lookup_green_taxi_01_2016")
    exit()

# Read DataSet
dataSet_path = "./data/"
df = fn.read_dataSet(dataSet_path)
lookup_table = {}

# Clean Columns
df = fn.clean_column_names(df, rename_dict={"lpep pickup datetime": "pickup_datetime", "lpep dropoff datetime":"dropoff_datetime","store and fwd flag":"store_and_fwd","pu_location":"pickup_location","do_location":"dropoff_location"})
df = fn.convert_to_datetime(df, ['pickup_datetime', 'dropoff_datetime'])
df.drop_duplicates(inplace=True)

# Make a copy of the Dataframe
df_copy = df.copy()

# Handle Missing Values
df_copy.drop(columns=['ehail_fee', 'congestion_surcharge'], inplace=True)
fn.statistical_imputation(lookup_table, df_copy, "passenger_count", method="mode")
fn.fill_missing_with_zeros(lookup_table, df_copy, "extra")
df_copy.loc[df_copy['tip_amount'] != 0, 'payment_type'] = df_copy.loc[df_copy['tip_amount'] != 0, 'payment_type'].fillna('Credit card')
df_copy["payment_type"].replace("Uknown", "unknown", inplace=True) 
fn.impute_missing_with_category(lookup_table, df_copy, "payment_type", "unknown")
fn.impute_specific_value(lookup_table, df_copy, "payment_type",'unknown', method="mode")

# Handle Incorrect/Inconsistent Values
df_copy["passenger_count"] = df_copy["passenger_count"].astype(int)
valid_values = [0.5, 0.0, -0.5, 1, -1, 83.00]
df_copy = fn.adjust_column_for_multiple_values(df_copy, 'extra', 'total_amount', valid_values)
valid_values = [0.5, 0.0, -0.5]
df_copy = fn.adjust_column_for_multiple_values(df_copy, 'mta_tax', 'total_amount', valid_values)
valid_values = [0.3, 0.0, -0.3]
df_copy = fn.adjust_column_for_multiple_values(df_copy, 'improvement_surcharge', 'total_amount', valid_values)
df_copy = fn.remove_rows_with_value(df_copy, 'trip_type', 'Unknown')
df_copy = fn.adjust_column_value_based_on_condition(df_copy, 'improvement_surcharge', 0.3, 'trip_type', 'Street-hail')
df_copy = fn.adjust_column_value_based_on_condition(df_copy, 'improvement_surcharge', -0.3, 'trip_type', 'Street-hail')
condition = (df_copy['tip_amount'] > 0) & (df_copy['payment_type'] != "Credit card")
df_copy.loc[condition, 'payment_type'] = 'Credit card'
df_copy = fn.remove_mismatched_rows(df_copy,['extra', 'mta_tax', 'tolls_amount','improvement_surcharge','fare_amount','tip_amount'], 'total_amount')
remove_pairs = [('total_amount', 0), ('trip_distance', 0)]
df_copy = fn.remove_rows_with_multiple_conditions(df_copy, remove_pairs)
conditions = [('vendor', '==', 'Creative Mobile Technologies, LLC'), ('total_amount', '==', 0)]
adjustments = {'rate_type': 'Negotiated fare', 'payment_type': 'No charge','trip_type':'Dispatch'}
df_copy = fn.adjust_features_based_on_conditions(df_copy, conditions, adjustments)
remove_pairs = [('total_amount', 0), ('vendor',"VeriFone Inc." )]
df_copy = fn.remove_rows_with_multiple_conditions(df_copy, remove_pairs)
conditions = [('rate_type', '==', 'Group ride'), ('passenger_count', '==', 1)]
adjustments = {'passenger_count': 2}
df_copy = fn.adjust_features_based_on_conditions(df_copy, conditions, adjustments)
df_copy = fn.replace_values(lookup_table, df_copy, 'trip_distance', 0, -1)

start_date = "2016-01-01"
end_date = "2016-01-31"
df_copy = fn.filter_records_by_date_range(df_copy, "pickup_datetime", start_date, end_date)

# Handle Outliers
df_copy = fn.fix_unwanted_values(df_copy, "passenger_count", 666, 6)
valid_values = [0.5, 0.0, 1 , -0.5 , -1]
df_copy = fn.adjust_column_for_multiple_values(df_copy, 'extra', 'total_amount', valid_values)
df_copy = fn.impute_outliers_with_mean_iqr(df_copy, 'trip_distance')
df_copy = fn.impute_outliers_with_mean_iqr(df_copy, 'fare_amount', 'total_amount')
df_copy = fn.impute_outliers_with_mean_iqr(df_copy, 'tip_amount', 'total_amount')
df_copy = fn.impute_outliers_with_mean(df_copy, 'tolls_amount','total_amount', 10)

# Discretization
df_copy = fn.equal_width_discretization(df_copy, "trip_distance", bins=5, new_column_name="trip_distance_discretized", labels=["low", "medium-low", "medium", "medium-high", "high"])
df_copy = fn.equal_width_discretization(df_copy, "fare_amount", bins=5, new_column_name="fare_amount_discretized", labels =["low", "medium-low", "medium", "medium-high", "high"])
df_copy = fn.equal_width_discretization(df_copy, "tip_amount", bins=3, new_column_name="tip_amount_discretized", labels =["low", "medium","high"])
df_copy = fn.equal_width_discretization(df_copy, "total_amount", bins=5, new_column_name="total_amount_discretized", labels =["low", "medium-low", "medium", "medium-high", "high"])
df_copy = fn.equal_width_discretization(df_copy, "pickup_datetime", bins=5, new_column_name="week_number")
df_copy = fn.equal_width_discretization_range(df_copy, "pickup_datetime", bins=5, new_column_name="date_range")

# Adding GPS
api_key = "AIzaSyBXV_Q4_CWvV7btH9drTwc3BYRoj2GwozQ"
coordinates_df = fn.gather_and_save_unique_coordinates(df_copy, api_key)
df_copy = fn.populate_lat_long(df_copy, 'pickup', coordinates_df)
df_copy = fn.populate_lat_long(df_copy, 'dropoff', coordinates_df)

# Encoding
onehot_features = ['vendor', 'store_and_fwd', 'payment_type', 'rate_type', 'trip_type']
label_features = ['pickup_location','dropoff_location']
df_copy, onehot_mappings = fn.encode_features(df_copy, onehot_features, 'onehot')
label_features = ['pickup_location','dropoff_location']
df_copy, label_mappings = fn.encode_features(df_copy, label_features, 'label')

# Lookup Table & Feature Engineering
fn.update_lookup(label_mappings, lookup_table)
df_copy = fn.add_features(df_copy)
flattened_data = []
for feature, mapping in lookup_table.items():
    for original_value, encoded_value in mapping.items():
        flattened_data.append({
            'Column Name': feature,
            'Original Value': original_value,
            'Imputed/Encoded Value': encoded_value
        })

# Save the Lookup Table
lookup_df = pd.DataFrame(flattened_data)
lookup_df.to_csv('./data/lookup_table_green_taxis.csv', index=False)

# Save the Cleaned DataSet
df_copy.to_csv('./data/green_tripdata_2016-01_clean.csv', index=False, header=True)

# Connect to the Database
ing.connectDB("./data/green_tripdata_2016-01_clean.csv", "./data/lookup_table_green_taxis.csv","green_taxi_01_2016", "lookup_green_taxi_01_2016")
