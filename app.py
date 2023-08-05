# 1. Library imports
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

class Attributes(BaseModel):
    GENDER: float
    AGE: float
    SMOKING: float
    YELLOW_FINGERS: float
    ANXIETY: float
    PEER_PRESSURE: float
    CHRONIC_DISEASE: float  # No space in the variable name
    WHEEZING: float
    ALCOHOL_CONSUMING: float  # No space in the variable name
    COUGHING: float
    SHORTNESS_OF_BREATH: float
    SWALLOWING_DIFFICULTY: float
    CHEST_PAIN: float

# 2. Create the app object
app = FastAPI()

# Load the trained classifier
with open("lung_cancer.pkl", "rb") as pickle_in:
    classifier = pickle.load(pickle_in)

# 3. Index route
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To TheManas Era': f'{name}'}

# 5. Prediction route
@app.post('/predict')
def predict_lung_cancer(data: Attributes):
    data_dict = data.dict()
    prediction = classifier.predict([
        [
            data_dict['GENDER'], data_dict['AGE'], data_dict['SMOKING'],
            data_dict['YELLOW_FINGERS'], data_dict['ANXIETY'],
            data_dict['PEER_PRESSURE'], data_dict['CHRONIC_DISEASE'],
            data_dict['WHEEZING'], data_dict['ALCOHOL_CONSUMING'],
            data_dict['COUGHING'], data_dict['SHORTNESS_OF_BREATH'],
            data_dict['SWALLOWING_DIFFICULTY'], data_dict['CHEST_PAIN']
        ]
    ])

    if prediction[0] == "1":
        prediction_text = "Lung Cancer Detected"
    else:
        prediction_text = "Chill Babe, it's normal"
    
    return {'prediction': prediction_text}

# 6. Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
