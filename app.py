from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Property(BaseModel):
    area: int
    property_type: str
    rooms_number: int
    zip_code: int
    land_area: Optional[int] = None
    garden: Optional[bool] = None
    garden_area: Optional[int] = None
    equipped_kitchen: Optional[bool] = None
    full_address: Optional[str] = None
    swimming_pool: Optional[bool] = None
    furnished: Optional[bool] = None
    open_fire: Optional[bool] = None
    terrace: Optional[bool] = None
    terrace_area: Optional[int] = None
    facades_number: Optional[int] = None
    building_state: Optional[str] = None

@app.get("/")
def read_root():
    return "alive"

@app.get("/predict")
def explain_predict():
    return"""
    Use this endpoint with the following data in JSON:
    
    - area: int *
    - property_type: str *
    - rooms_number: int *
    - zip_code: int *
    - land_area: int 
    - garden: bool 
    - garden_area: int 
    - equipped_kitchen: bool 
    - full_address: str 
    - swimming_pool: bool 
    - furnished: bool 
    - open_fire: bool 
    - terrace: bool 
    - terrace_area: int 
    - facades_number: int 
    - building_state: str 

    * Mandatory field
    """

@app.post("/predict")
def predict_property_price(property: Property):

    prediction = None 

    if prediction is not None:
        return {"prediction": prediction}
    else:
        return {"error": "Could not make a prediction, please try again."}

