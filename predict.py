# predict.py
import sys, json, joblib, os
import pandas as pd
import numpy as np
from scipy.sparse import hstack

# ---------------- Load Artifacts ----------------
ARTIFACTS_DIR = "C:/Project_edu2job/artifacts"

model = joblib.load(os.path.join(ARTIFACTS_DIR, "job_role_model.pkl"))
vectorizer = joblib.load(os.path.join(ARTIFACTS_DIR, "vectorizer.pkl"))
scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"))

# ---------------- Handle Input ----------------
if len(sys.argv) > 1:
    data = json.loads(sys.argv[1])   # input from Node.js
else:
    data = {
        "CGPA": 8.5,
        "Degree": "B.Tech",
        "Major": "Computer Science",
        "Skills": "Python, SQL, Machine Learning",
        "Certifications": "AWS, Azure",
        "Experience": 2
    }

df = pd.DataFrame([data])

# ---------------- Preprocessing (same as training) ----------------
# Normalize text
for col in ["Degree", "Major", "Skills", "Certifications"]:
    df[col] = df[col].astype(str).str.lower().str.strip()

# Derived features
df["num_skills"] = df["Skills"].apply(lambda x: 0 if x=="none" else len([s.strip() for s in x.split(",") if s.strip()]))
df["num_certs"] = df["Certifications"].apply(lambda x: 0 if x=="none" else len([c.strip() for c in x.split(",") if c.strip()]))
df["cgpa_x_exp"] = df["CGPA"] * df["Experience"]
df["skills_x_certs"] = df["num_skills"] * df["num_certs"]

# Profile text
df["profile_text"] = df["Degree"] + " " + df["Major"] + " " + df["Skills"] + " " + df["Certifications"]

# Vectorize text
X_text = vectorizer.transform(df["profile_text"])

# Scale numeric
numeric_features = df[["CGPA", "Experience", "num_skills", "num_certs", "cgpa_x_exp", "skills_x_certs"]].values
numeric_scaled = scaler.transform(numeric_features)

# Final feature set
X = hstack([X_text, numeric_scaled])

# ---------------- Prediction ----------------
probs = model.predict_proba(X)[0]  # probability distribution
top3_idx = np.argsort(probs)[::-1][:3]
top3_roles = label_encoder.inverse_transform(top3_idx)
top3_conf = probs[top3_idx] * 100

results = [
    {"role": role, "confidence": round(conf, 2)}
    for role, conf in zip(top3_roles, top3_conf)
]

print(json.dumps(results))  # return JSON to Node.js
