# <YOUR_IMPORTS>
import os
import dill
import json
import pandas as pd
from datetime import datetime

# path = 'C:/Users/zavil/airflow_hw'
path = os.environ.get('PROJECT_PATH', '.')


def get_model():
    file_name = os.listdir(f'{path}/data/models')[0]
    with open(f'{path}/data/models/{file_name}', 'rb') as file:
        model = dill.load(file)
    return model


def predict():
    df_result = pd.DataFrame(columns=['car_id', 'pred'])
    for file in os.listdir(f'{path}/data/test'):
        with open(f'{path}/data/test/{file}') as f:
            car_dict = json.load(f)
            car_df = pd.DataFrame.from_dict([car_dict])
            y = get_model().predict(car_df)
            df_pred = pd.DataFrame({'car_id': [car_dict['id']], 'pred': [y[0]]})
            df_result = pd.concat([df_result, df_pred], axis=0, ignore_index=True)
    df_result.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)


if __name__ == '__main__':
    predict()

