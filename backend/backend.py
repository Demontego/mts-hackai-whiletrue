from fastapi import FastAPI, Form
from pydantic import BaseModel
import uvicorn
import numpy as np
import os, json, random
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

test = [{'itemid': '10', 'prob': '0.9999', 'price': 1000000}, 
        {'itemid': '12', 'prob': '0.5', 'price': 1123}]



@app.post('/getdata')
async def get_id(ID: str = Form(...)):
    print(ID)
    return {'body': test, 'status': 'error'}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8010)
