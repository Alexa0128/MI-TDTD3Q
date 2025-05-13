
# hitelkartya_csalas_detektalas.py
# Hitelkártyás csalások detektálása gépi tanulással (Random Forest + SMOTE)
# Adathalmaz: creditcard.csv

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Adatok betöltése
print("Adatok betöltése...")
df = pd.read_csv("creditcard.csv")
print(f"Sorok száma: {df.shape[0]}, Oszlopok száma: {df.shape[1]}")

# 2. Alap statisztika
print("\nOsztályeloszlás:")
print(df['Class'].value_counts())

# 3. Adatok előkészítése
X = df.drop("Class", axis=1)
y = df["Class"]

# 4. Tanító és teszt adathalmaz szétválasztása
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. SMOTE alkalmazása (kiegyensúlyozás)
print("\nSMOTE alkalmazása a tanító adatokra...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"Tanító adatok mérete SMOTE után: {X_train_res.shape}")

# 6. Random Forest modell tanítása
print("\nModell tanítása...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_res, y_train_res)

# 7. Előrejelzés és értékelés
print("\nElőrejelzés és kiértékelés...")
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC AUC
roc_score = roc_auc_score(y_test, y_proba)
print(f"ROC AUC Score: {roc_score:.4f}")

# 8. Confusion Matrix vizualizálása
print("\nKonfúziós mátrix megjelenítése...")
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

print("\nA program lefutott. Az eredmények mentésre kerültek.")
