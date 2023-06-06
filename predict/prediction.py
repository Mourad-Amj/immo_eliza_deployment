from typing import Dict
from model.model import load_model

def predict_price(property_type: str, property: Dict[str, any]) -> float:
    model = load_model(property_type)

    property_data = {
        'Living area': property['area'],
        'Type of property': property['property_type'],
        'Number of rooms': property['rooms_number'],
        'Zip': property['zip_code'],
        'Surface of the land': property.get('land_area'),
        'garden': property.get('garden'),
        'Garden surface': property.get('garden_area'),
        'Kitchen values': property.get('equipped_kitchen'),   
        'Swimming pool': property.get('swimming_pool'),
        'Furnished': property.get('furnished'),
        'Open fire': property.get('open_fire'),
        'Terrace': property.get('terrace'),
        'Terrace surface': property.get('terrace_area'),
        'Number of facades': property.get('facades_number'),
        'Building Cond. values': property.get('building_state')
    }
    prediction = model.predict_price(property_data)
    # prediction = model.predict_price(property_data)
    return prediction
