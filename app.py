import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="ArvyaX Emotional Guidance API")

# Setup templates
os.makedirs("templates", exist_ok=True)
templates = Jinja2Templates(directory="templates")

# Try loading the model globally
try:
    # We must have the class ArvyaxModel & DecisionEngine accessible to unpickle. 
    # For a simple demo, we load the pipeline module which has the class definition.
    import pipeline
    model = joblib.load('arvyax_model.joblib')
    engine = pipeline.DecisionEngine()
    print("Successfully loaded the ArvyaX model locally.")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None
    engine = None

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "prediction": None
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict_reflection(
    request: Request,
    journal_text: str = Form(""),
    ambience_type: str = Form("none"),
    duration_min: int = Form(10),
    sleep_hours: float = Form(7.0),
    energy_level: int = Form(3),
    stress_level: int = Form(3),
    time_of_day: str = Form("afternoon"),
    previous_day_mood: str = Form("neutral"),
    face_emotion_hint: str = Form("none"),
    reflection_quality: str = Form("vague")
):
    if not model:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Model not loaded. Please run 'python pipeline.py' first to train and save the model."
        })
        
    # Create single row dataframe matching train features
    input_data = pd.DataFrame([{
        'id': 9999,
        'journal_text': journal_text,
        'ambience_type': ambience_type,
        'duration_min': duration_min,
        'sleep_hours': sleep_hours,
        'energy_level': energy_level,
        'stress_level': stress_level,
        'time_of_day': time_of_day,
        'previous_day_mood': previous_day_mood,
        'face_emotion_hint': face_emotion_hint,
        'reflection_quality': reflection_quality
    }])
    
    try:
        # Predict using model
        states, intensities, probs = model.predict(input_data)
        state = states[0]
        intensity = intensities[0]
        
        # Reason what to do
        what, when = engine.decide(state, intensity, stress_level, energy_level, time_of_day)
        
        # Uncertainty
        from pipeline import calculate_uncertainty
        conf, uncert = calculate_uncertainty(probs[0])
        
        # Generative Support Message
        support_msg = engine.generate_message(state, intensity, what, when)
        
        prediction = {
            "state": state,
            "intensity": int(intensity),
            "confidence": f"{conf:.1%}",
            "is_uncertain": bool(uncert),
            "what_to_do": what.replace("_", " ").title(),
            "when_to_do": when.replace("_", " ").title(),
            "support_message": support_msg
        }
        
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Error during prediction: {str(e)}"
        })

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": prediction,
        "input_text": journal_text
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
# Reload trigger
