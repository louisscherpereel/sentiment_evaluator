# Emotion Evaluator

## Overview
This project evaluates customer review sentiment using pre-trained NLP models.
You can run it as:

- A batch Python script (`main.py`) producing a CSV output  
- A **Streamlit demo** (`app.py`)  
- A **FastAPI REST API** (`api.py`) for programmatic access

## Models Used
- VADER (rule-based)
- DistilBERT (transformer-based)

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run the project

### Batch script
```bash
python src/main.py
```
To get a benchmark evaluation of both models

### Streamlit demo
```bash
streamlit run src/app.py
```
Open browser at http://localhost:8501


### Fast API endpoint
```bash
uvicorn src.api:app --reload
```
Access Swagger UI on http://127.0.0.1:8000/docs


## Possible next steps
- Add more pre-trained models
- Implement multi-class sentiment
- Expose batch prediction endpoint
- Dockerize for production deployment
