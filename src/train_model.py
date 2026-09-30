import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/ai4i2020.csv")
features = ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
target = "Machine failure"
X, y = df[features], df[target]
pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), ["Type"]),
    ("num", StandardScaler(), [c for c in features if c != "Type"])
])
model = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1
)
pipe = Pipeline([("preprocessor", pre), ("model", model)])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.20, stratify=y, random_state=42
)
pipe.fit(X_train, y_train)
joblib.dump(pipe, "models/machine_failure_model.joblib")
print("Model saved.")
