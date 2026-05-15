# Mini Power BI + AutoML Project

## Features
- FastAPI backend for CSV upload and model training
- Streamlit frontend for dashboard and AutoML
- Automatic classification/regression detection
- Plotly visualizations

## Project Structure
- backend/main.py
- frontend/app.py
- data/sample_sales.csv
- requirements.txt

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run Backend

```bash
cd backend
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open API docs at http://127.0.0.1:8000/docs

## Run Frontend

```bash
cd frontend
streamlit run app.py
```

Open Streamlit at http://localhost:8501