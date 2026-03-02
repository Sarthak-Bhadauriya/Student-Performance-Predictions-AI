# Student Performance Predictions AI

## About This Project
Hi, I am **Sarthak Bhadauriya**. I built this project to predict student performance using Machine Learning and serve predictions through a Flask web application.

My goal with this project was simple:
- train a reliable ML model on student data
- provide an easy UI for single and bulk predictions
- keep track of past predictions for review

## What I Built
- Flask-based web app for prediction workflow
- Trained ML model with preprocessing pipeline
- Single prediction form (UI)
- Bulk prediction support using CSV/Excel input
- Prediction history page (`/history`)
- SQLite database integration for storing records

## Tech Stack
- Python
- Flask
- scikit-learn
- pandas / numpy
- SQLite
- HTML templates

## Project Structure
```text
Main-student-perfomance/
|- app/
|  |- app.py
|  |- routes.py
|  |- templates/
|  |- model/
|  `- utils/
|- data/
|- train_model/
|- requirements.txt
|- Procfile
`- README.md
```

## How To Run Locally
1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Train model
```bash
python train_model/train.py
```

3. Start Flask app
```bash
python app/app.py
```

4. Open in browser
```text
http://127.0.0.1:5000
```

## Usage
- Use the home page to enter student details and get performance prediction.
- Upload CSV/Excel for bulk predictions.
- Open `/history` to view previous prediction logs.

## Why This Project Matters To Me
This project reflects my interest in combining:
- data-driven decision systems
- practical web development
- end-to-end ML deployment

## Future Improvements
- better model experimentation and tuning
- user authentication for private dashboards
- cloud deployment with CI/CD
- advanced analytics and visualization

## Author
**Sarthak Bhadauriya**
