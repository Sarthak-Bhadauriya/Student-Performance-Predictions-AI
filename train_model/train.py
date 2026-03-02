import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1️⃣ Load dataset with new columns
df = pd.read_csv('../data/dataset.csv')

# 2️⃣ Fill missing values (if any)
df.fillna(method='ffill', inplace=True)

# 3️⃣ Encode categorical columns
label_encoders = {}
categorical_cols = ['Gender', 'FinalGrade']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 4️⃣ Select features and target
X = df[['Attendance', 'AssignmentScore', 'MidtermMarks', 'Gender']]  # Only 4 features
y = df['FinalGrade']

# 5️⃣ Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7️⃣ Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8️⃣ Evaluation
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# 9️⃣ Save model/artifacts
joblib.dump(model, '../app/model/model.pkl')
joblib.dump(scaler, '../app/model/scaler.pkl')
joblib.dump(label_encoders, '../app/model/label_encoders.pkl')

print("✅ Model & preprocessors saved in app/model/")