import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import numpy as np

# Step 1: Load and Prepare the Data
df = pd.read_csv("C:\\Users\\mouni\\Downloads\\CAPSTONE PROJECT 2\\diet_recommendations_expanded.csv")

# If 'Severity' column doesn't exist, add it with random values or rules
if 'Severity' not in df.columns:
    # Simulate Severity based on Disease_Type (e.g., 'None' gets 'none', others get random severity)
    severity_options = ['Mild', 'Moderate', 'Severe']
    df['Severity'] = df['Disease_Type'].apply(
        lambda x: 'none' if x == 'None' else np.random.choice(severity_options)
    )

# Step 2: Calculate Recommended Calories (Rule-Based) as Target
def calculate_recommended_calories(row):
    # BMR (Mifflin-St Jeor)
    if row['Gender'] == 'Male':
        bmr = 10 * row['Weight_kg'] + 6.25 * row['Height_cm'] - 5 * row['Age'] + 5
    else:
        bmr = 10 * row['Weight_kg'] + 6.25 * row['Height_cm'] - 5 * row['Age'] - 161
    
    # TDEE
    activity_factors = {'Sedentary': 1.2, 'Moderate': 1.375, 'Active': 1.55}
    tdee = bmr * activity_factors[row['Physical_Activity_Level']]
    
    # Adjust for Disease Type and Severity
    severity_factors = {'Mild': 1.05, 'Moderate': 1.0, 'Severe': 0.95, 'none': 1.0}
    severity_adjustment = severity_factors.get(row['Severity'], 1.0)
    
    if row['Disease_Type'] == 'Obesity':
        rec_cal = max(tdee - 750, 1500 if row['Gender'] == 'Male' else 1200) * severity_adjustment
    elif row['Disease_Type'] == 'Diabetes':
        rec_cal = (tdee - 300 if row['BMI'] > 25 else tdee) * severity_adjustment
    elif row['Disease_Type'] == 'Hypertension':
        rec_cal = (tdee - 300 if row['BMI'] > 25 else tdee) * severity_adjustment
    else:  # 'None'
        rec_cal = tdee * severity_adjustment
    
    return round(rec_cal)

df['Recommended_Calories'] = df.apply(calculate_recommended_calories, axis=1)

# Step 3: Encode Categorical Variables
# Ensure 'None' is a valid label for Disease_Type
df['Disease_Type'] = df['Disease_Type'].astype(str)
if 'None' not in df['Disease_Type'].unique():
    dummy_row = df.iloc[0].copy()
    dummy_row['Disease_Type'] = 'None'
    dummy_row['Severity'] = 'none'
    df = pd.concat([df, pd.DataFrame([dummy_row])], ignore_index=True)

# Initialize and fit LabelEncoders
gender = LabelEncoder()
disease = LabelEncoder()
activity = LabelEncoder()
severity = LabelEncoder()

df['Gender'] = gender.fit_transform(df['Gender'])
df['Disease_Type'] = disease.fit_transform(df['Disease_Type'])
df['Physical_Activity_Level'] = activity.fit_transform(df['Physical_Activity_Level'])
df['Severity'] = severity.fit_transform(df['Severity'])

# Save the label encoders
with open('gender.pkl', 'wb') as f:
    pickle.dump(gender, f)
with open('disease.pkl', 'wb') as f:
    pickle.dump(disease, f)
with open('activity.pkl', 'wb') as f:
    pickle.dump(activity, f)
with open('severity.pkl', 'wb') as f:
    pickle.dump(severity, f)

# Step 4: Train the Model
X = df[['Age', 'Gender', 'Weight_kg', 'Height_cm','Disease_Type', 'Severity','Physical_Activity_Level']]
y = df['Recommended_Calories']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"R^2 Score: {r2}")

# Step 5: Save the Model
with open('chatbot.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model and label encoders saved as pickle files.")
print("Encoder classes:")
print(f"Gender: {gender.classes_}")
print(f"Disease_Type: {disease.classes_}")
print(f"Physical_Activity_Level: {activity.classes_}")
print(f"Severity: {severity.classes_}")