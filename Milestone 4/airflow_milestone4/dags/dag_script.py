from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator

import Task1 as t1
import Task2 as t2
import Task3 as t3
import Task4 as t4

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    'start_date': days_ago(2),
    "retries": 1,
}

dag = DAG(
    'green_taxis_pipeline_2016-01',
    default_args=default_args,
    description='green_taxis_pipeline_2016-01',
)
with DAG(
    dag_id = 'green_taxis_pipeline_2016-01',
    schedule_interval = '@once',
    default_args = default_args,
    tags = ['green_taxis_pipeline_2016-01'],
)as dag:
    read_clean_transform_load_task= PythonOperator(
        task_id = 'read_clean_transform_load_task',
        python_callable = t1.read_clean_transform_load_task,
        op_kwargs={
            "filename": './data/green_tripdata_2016-01.csv'
        },
    )
    extract_additional_resources_task= PythonOperator(
        task_id = 'extract_additional_resources_task',
        python_callable = t2.extract_additional_resources_task,
        op_kwargs={
            "filename": "./data/green_tripdata_2016-01_beforeEncoding.csv"
        },
    )
    integrate_load_to_postgres_csv_task=PythonOperator(
        task_id = 'integrate_load_to_postgres_csv_task',
        python_callable = t3.integrate_load_to_postgres_csv_task,
        op_kwargs={
            "gps_file_path": "./data/all_location_coordinates.csv",
            "afterEncoding_file_path": "./data/green_tripdata_2016-01_task1.csv",
            "lookup_file_path": "./data/lookup_table_green_taxis.csv"
        },
    )
    create_dashboard_task= PythonOperator(
        task_id = 'create_dashboard_task',
        python_callable = t4.create_dashboard_task,
        op_kwargs={
            "filename": "./data/green_tripdata_2016-01_beforeEncoding.csv",
        },
    )
    


    read_clean_transform_load_task >> extract_additional_resources_task >> integrate_load_to_postgres_csv_task >> create_dashboard_task

    
    



