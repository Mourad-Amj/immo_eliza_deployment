from typing import Optional, Literal
from fastapi import FastAPI
from pydantic import BaseModel
from predict.prediction import predict   

app = FastAPI()

class Property(BaseModel):
    area: int
    property_type: Literal["APARTMENT", "HOUSE"]
    rooms_number: int
    zip_code: int
    land_area: Optional[int]
    garden: Optional[bool]
    garden_area: Optional[int]
    equipped_kitchen: Optional[Literal['Not installed', 'Installed', 'Semi equipped', 'Hyper equipped', 'USA uninstalled',
                                        'USA installed', 'USA semi equipped', 'USA hyper equipped']]
    full_address: Optional[str]
    swimming_pool: Optional[bool]
    furnished: Optional[bool]
    open_fire: Optional[bool]
    terrace: Optional[bool]
    terrace_area: Optional[int]
    facades_number: Optional[int]
    building_state: Optional[Literal['To restore', 'To be done up', 'Just renovated', 'To renovate', 'Good', 'As new']]

@app.get("/")
def read_root():
    return "alive"

@app.get("/predict")
def explain_predict():
    return """
    Use this endpoint with the following data in JSON:
    
    - area: int *
    - property_type: str *
    - rooms_number: int *
    - zip_code: int *
    - land_area: int 
    - garden: bool 
    - garden_area: int 
    - equipped_kitchen: str 
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
    prediction = predict(property.property_type, property.dict())

    if prediction is not None:
        return {"prediction": prediction}
    else:
        return {"error": "Could not make a prediction, please try again."}
    




#     # load the model from disk
# loaded_model = joblib.load('/mnt/c/Users/Moura/git/immo_eliza_deployment/model/LM_model_house.sav')

# predictions = loaded_model.predict(X_test)
# result = loaded_model.score(X_test, y_test)
