# =====================================
# train_model.py (Random Forest Optimized for 80-90% Accuracy)
# =====================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack
import joblib

print("🔄 Loading dataset...")
df = pd.read_csv("C:\\Project_edu2job\\backend\\Education_Dataset.csv")
df.columns = df.columns.str.strip().str.lower()
print("✅ Dataset loaded:", df.shape)

# =====================================
# Step 1: Handle Missing Values
# =====================================
df['certifications'] = df['certifications'].fillna("None")
df['skills'] = df['skills'].fillna("None")
df['degree'] = df['degree'].fillna("Unknown")
df['major'] = df['major'].fillna("Unknown")
df['cgpa'] = df['cgpa'].fillna(df['cgpa'].mean())
df['experience'] = df['experience'].fillna(0)
print("✅ Missing values handled")

# =====================================
# Step 2: Normalize Text
# =====================================
for col in ["degree", "major", "skills", "certifications"]:
    df[col] = df[col].astype(str).str.lower().str.strip()
print("✅ Text normalized")

# =====================================
# Step 3: Derived Features
# =====================================
df["num_skills"] = df["skills"].apply(lambda x: 0 if x=="none" else len([s.strip() for s in x.split(",") if s.strip()]))
df["num_certs"] = df["certifications"].apply(lambda x: 0 if x=="none" else len([c.strip() for c in x.split(",") if c.strip()]))
# Interaction features
df["cgpa_x_exp"] = df["cgpa"] * df["experience"]
df["skills_x_certs"] = df["num_skills"] * df["num_certs"]
print("✅ Derived features & interactions created")

# =====================================
# Step 4: Profile Text
# =====================================
df["profile_text"] = df["degree"] + " " + df["major"] + " " + df["skills"] + " " + df["certifications"]
print("✅ Profile text created")

# =====================================
# Step 5: TF-IDF Vectorization
# =====================================
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1,3), min_df=1)
X_text = vectorizer.fit_transform(df["profile_text"])
print("TF-IDF shape:", X_text.shape)

# =====================================
# Step 6: Combine Text + Numeric Features
# =====================================
numeric_features = df[["cgpa", "experience", "num_skills", "num_certs", "cgpa_x_exp", "skills_x_certs"]].values
scaler = StandardScaler()
numeric_features = scaler.fit_transform(numeric_features)
X = hstack([X_text, numeric_features])
print("✅ Final feature matrix shape:", X.shape)

# =====================================
# Step 6b: Save Scaler
# =====================================
joblib.dump(scaler, "scaler.pkl")
print("✅ Scaler saved!")

# =====================================
# Step 7: Encode Target
# =====================================
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["job role"])
print("Classes:", label_encoder.classes_)

# =====================================
# Step 8: Train-Test Split
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train size:", X_train.shape, "Test size:", X_test.shape)

# =====================================
# Step 9: SMOTE Balancing
# =====================================
print("🔄 Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("✅ Balanced dataset:", X_train_res.shape, "→", len(np.unique(y_train_res)), "classes")

# =====================================
# Step 10: Train Random Forest
# =====================================
print("🔄 Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=1200,
    max_depth=50,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42
)
model.fit(X_train_res, y_train_res)
print("✅ Model trained")

# =====================================
# Step 11: Evaluation
# =====================================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

print("\n📊 Evaluation Results:")
print("Top-1 Accuracy:", accuracy_score(y_test, y_pred))
print("Top-3 Accuracy:", top_k_accuracy_score(y_test, y_proba, k=3))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Optional: cross-validation estimate
scores = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1)
print("Cross-validation mean accuracy:", scores.mean())

# =====================================
# Step 12: Save Artifacts
# =====================================
joblib.dump(model, "job_role_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("✅ Model, vectorizer, and label encoder saved!")