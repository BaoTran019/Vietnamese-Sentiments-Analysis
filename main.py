import pandas as pd
import numpy as np

data_folder_path = 'data'
dataset_path = data_folder_path + '/synthetic_train_converted_and_embedding.csv'

df_train = pd.read_csv(dataset_path)
print(df_train)