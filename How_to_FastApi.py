from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel, ValidationError

app = FastAPI()

class Data(BaseModel):
    salary: int
    bonus: int
    taxes: int

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/double/{num}")
def double(num: int):
    return {"result": num * 2}

@app.post("/compute")
def compute(data: Data):
    try:
        result = data.salary + data.bonus - data.taxes
        return {"result": result}
    except ValidationError as e:
        return {"error": "expected numbers, got strings."}