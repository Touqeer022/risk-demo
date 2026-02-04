import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Data Load Karein
try:
    df = pd.read_csv('german_credit_data.csv')
    print("✅ Data loaded successfully!")
except:
    print("❌ Error: 'german_credit_data.csv' nahi mili. File ka naam check karein.")
    exit()

# 2. Missing Values (Khali jagah) bharein
df['Saving accounts'] = df['Saving accounts'].fillna('little')
df['Checking account'] = df['Checking account'].fillna('little')

# 3. 'Risk' Column Banayen (Fixed for Numpy TypeError)
# Logic: Agar Credit Amount > 5000 hai ya Duration > 30 months hai, toh Risk = 'bad', warna 'good'
conditions = [
    (df['Credit amount'] > 5000) | (df['Duration'] > 30),
    (df['Credit amount'] <= 5000) & (df['Duration'] <= 30)
]
values = ['bad', 'good']

# FIX: Yahan default='good' add kiya hai taake error na aaye
df['Risk'] = np.select(conditions, values, default='good')

print("✅ Risk column created based on logic.")

# 4. Text ko Numbers mein badlein (Encoding)
encoders = {}
categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Risk']

# Data clean karein (convert all object columns to string to avoid mixed types)
for col in categorical_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# 5. Features (X) aur Target (y) alag karein
X = df[['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Purpose']]
y = df['Risk']

# 6. Model Train Karein
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
print("✅ Model trained successfully!")

# 7. Model aur Encoders ko Save karein
joblib.dump(model, 'credit_model.pkl')
joblib.dump(encoders, 'encoders.pkl')

print("🎉 Success! 'credit_model.pkl' file ban gayi hai.")
