import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

def feature_engineering(data):
  new_type = {'PAYMENT':'OTHERS', 'TRANSFER':'TRANSFER', 'CASH_OUT':'CASH_OUT', 'DEBIT':'OTHERS', 'CASH_IN':'OTHERS'}
  data['type']=data['type'].map(new_type)
  data['type2'] = np.nan
  data.loc[data.nameOrig.str.contains('C') & data.nameDest.str.contains('C'), 'type2'] = 'CC'
  data.loc[data.nameOrig.str.contains('C') & data.nameDest.str.contains('M'), 'type2'] = 'CM'
  data.loc[data.nameOrig.str.contains('M') & data.nameDest.str.contains('C'), 'type2'] = 'MC'
  data.loc[data.nameOrig.str.contains('M') & data.nameDest.str.contains('C'), 'type2'] = 'MM'
  data.drop(columns = ['nameOrig','nameDest'], axis = 'columns', inplace = True)
  return data

def preprocessing(data):
  data = pd.get_dummies(data, prefix = ['type', 'type2'], drop_first = False)
  column_order = ['step', 'amount', 'oldbalanceOrg','oldbalanceDest','newbalanceOrig','newbalanceDest','type_CASH_OUT',
       'type_OTHERS', 'type_TRANSFER', 'type2_CC', 'type2_CM','isFraud']
  data = data[column_order]
  return data

def sampling(df):
    sm = SMOTE(random_state=42)
    df_sm = sm.fit_sample(df, df['isFraud'])
    return df_sm

if __name__ == "__main__":
  df = pd.read_csv('/content/drive/MyDrive/Sanju Sarkar/dataset.csv')
  data = feature_engineering(df)
  data = preprocessing(data)
  data = sampling(data)
  data.to_csv('/content/drive/MyDrive/Sanju Sarkar/processed.csv')