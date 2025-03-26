from flask import Flask, render_template, request
import joblib
import pickle
import pandas as pd
import random

app = Flask(__name__)

# Load the trained model and label encoders
try:
    model = joblib.load('diet_recommendations_dataset.pkl')
    with open('le_gender.pkl', 'rb') as f:
        le_gender = pickle.load(f)
    with open('le_disease.pkl', 'rb') as f:
        le_severity = pickle.load(f)  # Actually severity: ['Mild', 'Moderate', 'Severe']
    with open('le_activity.pkl', 'rb') as f:
        le_activity = pickle.load(f)
    with open('le_severity.pkl', 'rb') as f:
        le_unused = pickle.load(f)  # Misnamed, currently activity
    print("Encoder classes loaded:")
    print("le_gender.classes_:", le_gender.classes_)
    print("le_severity.classes_ (loaded as le_disease):", le_severity.classes_)
    print("le_activity.classes_:", le_activity.classes_)
    print("le_unused.classes_ (loaded as le_severity):", le_unused.classes_)
except FileNotFoundError as e:
    raise Exception(f"Missing file: {str(e)}. Ensure all model and encoder files are present.")

# Manual mapping for disease types
disease_mapping = {'None': 0, 'diabetes': 1, 'hypertension': 2, 'obesity': 3}

# Function to predict caloric intake
def predict_caloric_intake(age, gender, weight, height, disease_type, severity, activity_level):
    try:
        print(f"predict_caloric_intake inputs: Age={age}, Gender={gender}, Weight={weight}, Height={height}, "
              f"Disease={disease_type}, Severity={severity}, Activity={activity_level}")

        gender_encoded = le_gender.transform([gender.capitalize()])[0]
        disease_encoded = disease_mapping.get(disease_type.lower(), 0)
        activity_encoded = le_activity.transform([activity_level.capitalize()])[0]
        severity_encoded = 0 if disease_type.lower() == 'none' else le_severity.transform([severity.capitalize()])[0]

        print(f"Encoded values: Gender={gender_encoded}, Disease={disease_encoded}, "
              f"Severity={severity_encoded}, Activity={activity_encoded}")

        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender_encoded],
            'Weight_kg': [weight],
            'Height_cm': [height],
            'Disease_Type': [disease_encoded],
            'Severity': [severity_encoded],
            'Physical_Activity_Level': [activity_encoded]
        })

        expected_columns = model.feature_names_in_
        for col in expected_columns:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[expected_columns]
        caloric_intake = model.predict(input_data)[0]
        return caloric_intake
    except ValueError as e:
        print(f"ValueError in predict_caloric_intake: {str(e)}")
        raise

# Function to recommend meals with variety
def recommend_meals(caloric_intake, disease_type, severity, diet_preference):
    # Proportional distribution: Breakfast 25%, Lunch 35%, Dinner 30%, Snacks 10%
    breakfast_calories = int(caloric_intake * 0.25)
    lunch_calories = int(caloric_intake * 0.35)
    dinner_calories = int(caloric_intake * 0.30)
    snacks_calories = int(caloric_intake * 0.10)

    # Severity adjustment for slight variation (not affecting total)
    severity_mapping = {'Mild': 1.05, 'Moderate': 1.0, 'Severe': 0.95, 'none': 1.0}
    adjustment = severity_mapping.get(severity.capitalize() if disease_type.lower() != 'none' else 'none', 1.0)

    # Food options for variety
    veg_breakfast_options = [
        f"Poha with peanuts and cucumber salad (approx. {int(breakfast_calories * adjustment)} calories)",
        f"Vegetable upma with mint chutney (approx. {int(breakfast_calories * adjustment)} calories)",
        f"Moong dal cheela with yogurt (approx. {int(breakfast_calories * adjustment)} calories)",
        f"Masala oats with chopped veggies (approx. {int(breakfast_calories * adjustment)} calories)"
    ]
    nonveg_breakfast_options = [
        f"Egg bhurji with mint chutney (approx. {int(breakfast_calories * adjustment)} calories)",
        f"Chicken omelette with whole wheat toast (approx. {int(breakfast_calories * adjustment)} calories)",
        f"Grilled fish with tomato salsa (approx. {int(breakfast_calories * adjustment)} calories)",
        f"Keema paratha with curd (approx. {int(breakfast_calories * adjustment)} calories)"
    ]

    veg_lunch_options = [
        f"Palak paneer with brown rice (approx. {int(lunch_calories * adjustment)} calories)",
        f"Rajma masala with jeera rice (approx. {int(lunch_calories * adjustment)} calories)",
        f"Chole with whole wheat roti (approx. {int(lunch_calories * adjustment)} calories)",
        f"Vegetable khichdi with curd (approx. {int(lunch_calories * adjustment)} calories)"
    ]
    nonveg_lunch_options = [
        f"Grilled chicken curry with brown rice (approx. {int(lunch_calories * adjustment)} calories)",
        f"Fish curry with quinoa (approx. {int(lunch_calories * adjustment)} calories)",
        f"Chicken tikka masala with naan (approx. {int(lunch_calories * adjustment)} calories)",
        f"Mutton rogan josh with steamed rice (approx. {int(lunch_calories * adjustment)} calories)"
    ]

    veg_dinner_options = [
        f"Grilled tofu tikka with sautéed vegetables (approx. {int(dinner_calories * adjustment)} calories)",
        f"Paneer bhurji with multigrain roti (approx. {int(dinner_calories * adjustment)} calories)",
        f"Stuffed capsicum with dal (approx. {int(dinner_calories * adjustment)} calories)",
        f"Vegetable pulao with raita (approx. {int(dinner_calories * adjustment)} calories)"
    ]
    nonveg_dinner_options = [
        f"Tandoori fish with steamed vegetables (approx. {int(dinner_calories * adjustment)} calories)",
        f"Chicken kebabs with mixed greens (approx. {int(dinner_calories * adjustment)} calories)",
        f"Grilled mutton chops with roasted veggies (approx. {int(dinner_calories * adjustment)} calories)",
        f"Prawn stir-fry with brown rice (approx. {int(dinner_calories * adjustment)} calories)"
    ]

    veg_snacks_options = [
        f"Roasted chana or cucumber sticks (approx. {int(snacks_calories * adjustment)} calories)",
        f"Sprouts salad with lemon (approx. {int(snacks_calories * adjustment)} calories)",
        f"Fruit chaat with yogurt (approx. {int(snacks_calories * adjustment)} calories)",
        f"Baked samosa with mint chutney (approx. {int(snacks_calories * adjustment)} calories)"
    ]
    nonveg_snacks_options = [
        f"Boiled eggs or grilled chicken strips (approx. {int(snacks_calories * adjustment)} calories)",
        f"Chicken tikka bites (approx. {int(snacks_calories * adjustment)} calories)",
        f"Fish fingers with tartar dip (approx. {int(snacks_calories * adjustment)} calories)",
        f"Keema patties (approx. {int(snacks_calories * adjustment)} calories)"
    ]

    # Select random options based on disease and diet preference
    if disease_type.lower() == 'diabetes':
        if diet_preference.lower() == 'veg':
            return {
                "breakfast": random.choice(veg_breakfast_options),
                "lunch": random.choice(veg_lunch_options),
                "dinner": random.choice(veg_dinner_options),
                "snacks": random.choice(veg_snacks_options)
            }
        else:
            return {
                "breakfast": random.choice(nonveg_breakfast_options),
                "lunch": random.choice(nonveg_lunch_options),
                "dinner": random.choice(nonveg_dinner_options),
                "snacks": random.choice(nonveg_snacks_options)
            }
    elif disease_type.lower() == 'hypertension':
        if diet_preference.lower() == 'veg':
            return {
                "breakfast": random.choice(veg_breakfast_options),
                "lunch": random.choice(veg_lunch_options),
                "dinner": random.choice(veg_dinner_options),
                "snacks": random.choice(veg_snacks_options)
            }
        else:
            return {
                "breakfast": random.choice(nonveg_breakfast_options),
                "lunch": random.choice(nonveg_lunch_options),
                "dinner": random.choice(nonveg_dinner_options),
                "snacks": random.choice(nonveg_snacks_options)
            }
    elif disease_type.lower() == 'obesity':
        if diet_preference.lower() == 'veg':
            return {
                "breakfast": random.choice(veg_breakfast_options),
                "lunch": random.choice(veg_lunch_options),
                "dinner": random.choice(veg_dinner_options),
                "snacks": random.choice(veg_snacks_options)
            }
        else:
            return {
                "breakfast": random.choice(nonveg_breakfast_options),
                "lunch": random.choice(nonveg_lunch_options),
                "dinner": random.choice(nonveg_dinner_options),
                "snacks": random.choice(nonveg_snacks_options)
            }
    else:  # Default for 'none' or unrecognized disease
        if diet_preference.lower() == 'veg':
            return {
                "breakfast": random.choice(veg_breakfast_options),
                "lunch": random.choice(veg_lunch_options),
                "dinner": random.choice(veg_dinner_options),
                "snacks": random.choice(veg_snacks_options)
            }
        else:
            return {
                "breakfast": random.choice(nonveg_breakfast_options),
                "lunch": random.choice(nonveg_lunch_options),
                "dinner": random.choice(nonveg_dinner_options),
                "snacks": random.choice(nonveg_snacks_options)
            }

# Flask routes
@app.route('/')
def home():
    print("Rendering index.html")
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        print("Received form data:", dict(request.form))
        age = int(request.form['age'])
        gender = request.form['gender']
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        disease_type = request.form['disease_type']
        severity_raw = request.form.get('severity', 'moderate') if disease_type.lower() != 'none' else 'none'
        severity = {'low': 'Mild', 'moderate': 'Moderate', 'high': 'Severe', 'none': 'none'}.get(severity_raw.lower(), 'Moderate')
        activity_level = request.form['activity_level']
        diet_preference = request.form['diet_preference']

        print(f"Processed inputs: Age={age}, Gender={gender}, Weight={weight}, Height={height}, "
              f"Disease={disease_type}, Severity={severity}, Activity={activity_level}, Diet={diet_preference}")

        caloric_intake = predict_caloric_intake(age, gender, weight, height, disease_type, severity, activity_level)
        meal_recommendation = recommend_meals(caloric_intake, disease_type, severity, diet_preference)

        print(f"Rendering result.html with caloric_intake={caloric_intake}")
        return render_template('result.html',
                               calories=caloric_intake,
                               breakfast=meal_recommendation['breakfast'],
                               lunch=meal_recommendation['lunch'],
                               dinner=meal_recommendation['dinner'],
                               snacks=meal_recommendation['snacks'])
    except ValueError as e:
        print(f"ValueError in recommend: {str(e)}")
        return render_template('index.html', error=str(e))
    except Exception as e:
        print(f"Exception in recommend: {str(e)}")
        return render_template('index.html', error=f"An error occurred: {str(e)}")

# Run the app
if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)