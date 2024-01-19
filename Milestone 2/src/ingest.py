import pandas as pd
from sqlalchemy import create_engine


def connectDB(dataSet_path, lookup_path, dataSet_name, lookup_table_name):
	dataSet_df = pd.read_csv(dataSet_path)
	lookup_df = pd.read_csv(lookup_path)
	engine = create_engine('postgresql://root:root@pgdatabase:5432/green_taxis')

	if(engine.connect()):
		print('connected succesfully')
	else:
		print('failed to connect')

	dataSet_df.to_sql(name = dataSet_name,con = engine,if_exists='fail')
	lookup_df.to_sql(name = lookup_table_name,con = engine,if_exists='fail')










