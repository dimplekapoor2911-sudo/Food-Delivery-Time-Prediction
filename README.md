# Food Delivery Time Prediction ⏳

## About the Project
This project uses machine learning to predict food delivery time based on delivery distance, weather, road traffic, and delivery partner details.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Random Forest Regressor
- Streamlit

## Model Performance
- Test R² Score: 0.812
- Mean Absolute Error: 3.24 minutes
- Root Mean Squared Error: 4.07 minutes

These metrics are from the current test evaluation. Predictions are estimates and may differ from actual delivery times.

## Run the Application
Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app/app.py
```

## Project Structure
- `Notebooks/` — Data analysis and model development
- `app/` — Streamlit application
- `data/` — Dataset files, if included
- `models/` — Saved model, if included
- `requirements.txt` — Project dependencies

## Disclaimer
This project is for educational purposes. Predictions are not guaranteed delivery times.
