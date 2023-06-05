from typing import Dict
from model.model import load_model

def predict_price(property: Dict[str, any]) -> float:
    model = load_model()  

    property_data = {
        'area': property['area'],
        'property_type': property['property_type'],
        'rooms_number': property['rooms_number'],
        'zip_code': property['zip_code'],
        'land_area': property.get('land_area'),
        'garden': property.get('garden'),
        'garden_area': property.get('garden_area'),
        'equipped_kitchen': property.get('equipped_kitchen'),
        'full_address': property.get('full_address'),
        'swimming_pool': property.get('swimming_pool'),
        'furnished': property.get('furnished'),
        'open_fire': property.get('open_fire'),
        'terrace': property.get('terrace'),
        'terrace_area': property.get('terrace_area'),
        'facades_number': property.get('facades_number'),
        'building_state': property.get('building_state')
    }

    prediction = model.predict_price(property_data)
    return prediction
