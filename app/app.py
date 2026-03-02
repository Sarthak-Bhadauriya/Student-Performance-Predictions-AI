from flask import Flask, request, render_template, redirect, url_for, send_file
import joblib
import pandas as pd
import os
import sqlite3
import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

app = Flask(__name__)
LAST_BULK_RESULTS = []

FEATURE_ORDER = ['Attendance', 'AssignmentScore', 'MidtermMarks', 'Gender']
NUMERIC_BENCHMARKS = {
    'Attendance': 75,
    'AssignmentScore': 70,
    'MidtermMarks': 70
}

def init_db():
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            attendance REAL,
            assignment_score REAL,
            midterm_marks REAL,
            gender TEXT,
            predicted_result TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "model.pkl")
scaler_path = os.path.join(BASE_DIR, "model", "scaler.pkl")
encoders_path = os.path.join(BASE_DIR, "model", "label_encoders.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoders = joblib.load(encoders_path)


def build_personalized_tips(attendance, assignment_score, midterm_marks):
    tips = []
    weak_metrics = []
    moderate_metrics = []
    metrics = {
        'Attendance': attendance,
        'Assignment Score': assignment_score,
        'Midterm Marks': midterm_marks
    }

    for metric_name, value in metrics.items():
        if value < 60:
            weak_metrics.append(metric_name)
        elif value < 75:
            moderate_metrics.append(metric_name)

    if weak_metrics:
        tips.append(
            f"High-risk area: {', '.join(weak_metrics)} is low. Focus on improving these in the next 2 weeks."
        )
    if moderate_metrics:
        tips.append(
            f"For better stability, move {', '.join(moderate_metrics)} into the 75+ range."
        )
    if attendance < 75:
        tips.append("Attendance is below target, so the overall risk level is higher.")
    if assignment_score < 70:
        tips.append("Improving assignment score can directly improve the final outcome.")
    if midterm_marks < 70:
        tips.append("Midterm marks are weak; focused revision and mock tests can help.")
    if not tips:
        tips.append("Strong profile: keep your consistency, current performance looks stable.")

    return tips[:4]


def build_ai_explanation(input_data, gender_str):
    importances = getattr(model, 'feature_importances_', None)
    if importances is None:
        return "Model explanation unavailable for current model type.", []

    ranked = sorted(
        zip(FEATURE_ORDER, importances),
        key=lambda item: item[1],
        reverse=True
    )
    top_features = ranked[:3]
    explanation_points = []
    importance_rows = []

    for feature_name, importance in top_features:
        pct = round(float(importance) * 100, 1)
        if feature_name == 'Gender':
            point = (
                f"{feature_name} contributes {pct}% to this prediction. Current input value is '{gender_str}'."
            )
            context_value = gender_str
        else:
            value = input_data[feature_name]
            benchmark = NUMERIC_BENCHMARKS[feature_name]
            diff_text = "above benchmark" if value >= benchmark else "below benchmark"
            point = (
                f"{feature_name} contributes {pct}% and current value is {value:.1f} ({diff_text}, target {benchmark}+)."
            )
            context_value = f"{value:.1f}"

        explanation_points.append(point)
        importance_rows.append({
            'feature': feature_name,
            'importance': pct,
            'value': context_value
        })

    explanation_summary = (
        "This prediction is primarily based on the top influential factors: " + " ".join(explanation_points)
    )
    return explanation_summary, importance_rows

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    gender_str = form_data['Gender'].capitalize()
    gender_encoded = label_encoders['Gender'].transform([gender_str])[0]
    attendance = float(form_data['Attendance'])
    assignment_score = float(form_data['AssignmentScore'])
    midterm_marks = float(form_data['MidtermMarks'])

    input_features = [
        attendance,
        assignment_score,
        midterm_marks,
        gender_encoded
    ]
    input_scaled = scaler.transform([input_features])
    pred_encoded = model.predict(input_scaled)[0]
    pred_result = label_encoders['FinalGrade'].inverse_transform([pred_encoded])[0]
    input_data = {
        'Attendance': attendance,
        'AssignmentScore': assignment_score,
        'MidtermMarks': midterm_marks
    }
    ai_tips = build_personalized_tips(attendance, assignment_score, midterm_marks)
    ai_explanation, importance_rows = build_ai_explanation(input_data, gender_str)

    # Save prediction in database (store gender string and decoded result)
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (name, attendance, assignment_score, midterm_marks, gender, predicted_result)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        form_data.get('Name'),
        attendance,
        assignment_score,
        midterm_marks,
        gender_str,
        pred_result
    ))
    conn.commit()
    conn.close()

    return render_template(
        'index.html',
        prediction_text=f"Predicted Final Grade: {pred_result}",
        ai_tips=ai_tips,
        ai_explanation=ai_explanation,
        importance_rows=importance_rows
    )

@app.route('/predict_bulk', methods=['POST'])
def predict_bulk():
    global LAST_BULK_RESULTS
    file = request.files['file']
    if not file:
        return "No file uploaded", 400

    filename = file.filename.lower()
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        df = pd.read_excel(file, engine='openpyxl')
    elif filename.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        return "Invalid file format, please upload a CSV or Excel file.", 400

    df['Gender'] = df['Gender'].str.capitalize()
    gender_encoded = label_encoders['Gender'].transform(df['Gender'])
    df['Gender'] = gender_encoded

    features = ['Attendance', 'AssignmentScore', 'MidtermMarks', 'Gender']
    features_scaled = scaler.transform(df[features])
    pred_encoded = model.predict(features_scaled)
    df['Predicted_Result'] = label_encoders['FinalGrade'].inverse_transform(pred_encoded)

    # Decode gender back for saving and display
    df['Gender'] = label_encoders['Gender'].inverse_transform(df['Gender'])

    # Save bulk predictions in database
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    for idx, row in df.iterrows():
        cursor.execute("""
            INSERT INTO predictions (name, attendance, assignment_score, midterm_marks, gender, predicted_result)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            row['Name'],
            float(row['Attendance']),
            float(row['AssignmentScore']),
            float(row['MidtermMarks']),
            row['Gender'],
            row['Predicted_Result']
        ))
    conn.commit()
    conn.close()

    results = df.to_dict(orient='records')
    LAST_BULK_RESULTS = results
    return render_template('index.html', bulk_results=results)

@app.route('/history')
def history():
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name, attendance, assignment_score, midterm_marks, gender, predicted_result, timestamp
        FROM predictions ORDER BY timestamp DESC
    """)
    records = cursor.fetchall()
    conn.close()
    return render_template('history.html', records=records)


@app.route('/history/clear', methods=['POST'])
def clear_history():
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
    return redirect(url_for('history'))


@app.route('/download_bulk_pdf', methods=['POST'])
def download_bulk_pdf():
    results = LAST_BULK_RESULTS
    if not results:
        return "No bulk prediction results available. Run bulk prediction first.", 400

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=24,
        rightMargin=24,
        topMargin=24,
        bottomMargin=24
    )
    styles = getSampleStyleSheet()
    elements = []

    title = Paragraph("Student Performance - Bulk Prediction Report", styles['Title'])
    subtitle = Paragraph(
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Total records: {len(results)}",
        styles['Normal']
    )
    elements.extend([title, Spacer(1, 8), subtitle, Spacer(1, 12)])

    columns = list(results[0].keys())
    table_data = [columns]
    for row in results:
        table_data.append([str(row.get(col, "")) for col in columns])

    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2563eb")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#f8fbff"), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#c7d6ea")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name='bulk_prediction_report.pdf',
        mimetype='application/pdf'
    )

if __name__ == '__main__':
    app.run(debug=True)
