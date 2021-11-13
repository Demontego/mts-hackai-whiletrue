from fastapi import FastAPI, Form
from pydantic import BaseModel
import uvicorn
import numpy as np
import pandas as pd
import pickle
import os, json, random
from deepctr_torch.models import DeepFM, xDeepFM
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
import torch
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

all_data = pd.read_csv('../all_data_for_lama.csv', index_col='ID',low_memory=False)
with open("../automl_rd.pkl", "rb") as f:
    model = pickle.load(f)
    
with open("../lbes.pkl", "rb") as f:
    lbes = pickle.load(f)
    
with open("../ss.pkl", "rb") as f:
    ss = pickle.load(f)
    
ffm = torch.load('../ffm.pt', map_location='cpu')

target = 'MCC_CODE'
sparse_features = ['ID', 'MCC_CODE','post_index', 'GENDER','REG_CODE', 'MARITAL_STATUS']
dense_features = ['PROD_TYPE', 'MM_IN_BANK', 'MM_W_CARD', 'AGE', 
       'EDUCATION_LEVEL', 'DEPENDANT_CNT', 'INCOME_MAIN_AMT']

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# test = [{'itemid': '10', 'prob': '0.9999', 'price': 1000000}, 
#         {'itemid': '12', 'prob': '0.5', 'price': 1123}]



@app.post('/getdata')
async def get_id(ID: str = Form(...)):
    ID = int(ID)
    if ID not in all_data.index:
        return {'body': test, 'status': 'error'}
    pred_data = pd.DataFrame(all_data.loc[ID]).T
    pred = model.predict(pred_data).data[:]
    ind = np.argpartition(pred, -10)[-10:]
    prob = pred[ind]
    classes = pred[ind]
    ff_data = []
    for ai in classes:
        ff_row={}
        for i in sparse_features:
            if i =='post_index':
                ff_row['post_index'] = data[str(ai)+'_loc'].iloc[0]
            else:
                ff_row[i] = data[i].iloc[0]
        for i in dense_features:
            if i == 'PROD_TYPE':
                ff_row['PROD_TYPE'] = data['type_card'].iloc[0]
            else:
                ff_row[i] = data[i].iloc[0]
        ff_row['MCC_CODE'] = ai
        ff_data.append(ff_row)
    ff_data = pd.DataFrame(ff_data)
    ff_data[sparse_features+target] = ff_data[sparse_features+target].fillna('-1', )
    for feat in sparse_features+target:
        ff_data[feat] = lbes[feat].transform(ff_data[feat])
    ff_data[dense_features] = ss['mms'].transform(ff_data[dense_features])
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=ff_data[feat].nunique(),embedding_dim=7)
                       for i,feat in enumerate(sparse_features+target)] + [DenseFeat(feat, 1,)
                      for feat in dense_features]
    feature_names = get_feature_names(fixlen_feature_columns)
    test_model_input = {name: ff_data[name] for name in feature_names}
    pred_ans = ffm.predict(test_model_input,batch_size=1)
    pred_ans = ss['ts'].inverse_transform(pred_ans)
    test = []
    for item_id,p,price in zip(classes,prob,pred_ans):
        row = {'itemid': str(item_id), 'prob': str(p), 'price': str(price)}
        test.append(row)
    return {'body': test, 'status': 'error'}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8010)
