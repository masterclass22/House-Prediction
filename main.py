from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Load the trained model
model = joblib.load('modal/HouseModel.pkl')

df = pd.read_csv('./static/Modified_Bengaluru_House_Data.csv')
locations = df['location'].dropna().unique().tolist()

# Initialize FastAPI
app = FastAPI()

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Define a Pydantic model for the input data
class RealEstateInput(BaseModel):
    location: str
    total_sqft: float
    bath: float
    bhk: int


@app.get("/", response_class=HTMLResponse)
async def serve_index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "locations": locations})

@app.post("/predict")
def predict_car_price(real_estate_input: RealEstateInput):
    # Convert the input data into a pandas DataFrame
    input_data = pd.DataFrame([{
        'location': real_estate_input.location,
        'total_sqft': real_estate_input.total_sqft,
        'bath': real_estate_input.bath,
        'bhk': real_estate_input.bhk
    }])
    print(input_data)
    # Make the prediction using the trained model
    try:
        predicted_price = model.predict(input_data)
    except Exception as e:
        return {"error": f"error: {str(e)}"}
    # Return the predicted price as a response
    return {"predicted_price": predicted_price[0]}
