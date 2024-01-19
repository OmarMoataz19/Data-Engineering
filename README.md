
# Project Overview
This project is centered around the analysis and enhancement of NYC taxi data, with a specific focus on both green and yellow taxi datasets. The primary objective was to refine the dataset for green taxis through organization, visualization, and preparation for subsequent analysis or machine learning. The workflow was encapsulated into user-friendly packages using Docker, ensuring portability and ease of use. Subsequently, the processed data was integrated into a PostgreSQL database for convenient access. Similar steps were applied to the yellow taxi data using PySpark. Task organization was streamlined using Airflow within Docker, enhancing the project's efficiency in data cleaning, modification, and augmentation.

## Milestone 1: Data Preparation and Exploration (Green Taxis)
The first milestone aimed to load a CSV file, conduct exploratory data analysis with visualization, extract additional data, and perform feature engineering and preprocessing. The dataset used pertained to NYC green taxis, with separate datasets available for each month. The code developed is reproducible and adaptable to different months or years. **Dataset Download:** [Download Green Taxis Dataset](https://drive.google.com/drive/folders/1t8nBgbHVaA5roZY4z3RcAG1_JMYlSTqu)

## Milestone 2: Docker Packaging and PostgreSQL Integration
In the second milestone, the objective was to package the code from milestone 1 into a Docker image for universal deployment. Additionally, the cleaned and prepared dataset, along with a lookup table, was loaded into a PostgreSQL database, serving as the project's data warehouse.

## Milestone 3: Preprocessing Yellow Taxis Data with PySpark
The third milestone focused on preprocessing the 'New York yellow taxis' dataset, mirroring the processes applied to green taxis in milestone 1. Basic data preparation and analysis were executed using PySpark, maintaining consistency with the chosen month and year (matching the green taxi dataset).
**Dataset Download:** [Download Yellow Taxis Dataset](https://drive.google.com/drive/folders/1t8nBgbHVaA5roZY4z3RcAG1_JMYlSTqu)

## Milestone 4: Airflow Orchestration of Tasks
For the fourth milestone, tasks from milestones 1 and 2 were orchestrated using Airflow within Docker. The primary emphasis was on the green dataset, and preprocessing was carried out using Pandas for simplicity. The tasks involved reading the CSV file for green taxis, cleaning and transforming the data, loading both the cleaned dataset and lookup table into CSV files, extracting additional resources (GPS coordinates), integrating them with the cleaned dataset, and finally loading both CSV files into a PostgreSQL database as two separate tables. This orchestration streamlined the entire process, enhancing project manageability and scalability.
