from typing import Dict
import joblib  

def predict_price(property_type: str, property: Dict[str, any]) -> float:

    if property.property_type == "APARTMENT":
        loaded_model = joblib.load('/mnt/c/Users/Moura/git/immo_eliza_deployment/model/LM_model_Apartment.sav')

        prediction = loaded_model.predict(X_test)

    elif property.property_type == "HOUSE":
         loaded_model = joblib.load('/mnt/c/Users/Moura/git/immo_eliza_deployment/model/LM_model_house.sav')

         prediction = loaded_model.predict(X_test)

    else:
        raise ValueError("It should be either 'APARTMENT' or 'HOUSE'.")

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

    return prediction


df=pd.DataFrame(property_data)
