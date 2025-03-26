from flask import Flask, render_template, request, session, jsonify
from datetime import timedelta
import pickle
import pandas as pd
import numpy as np
import random
import re

app = Flask(__name__)
app.secret_key = 'd1f123bd7246e06f01f198ac76f08f18'  # Change this!
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Load models and encoders
with open('chatbot.pkl', 'rb') as f:
    model = pickle.load(f)
with open('gender.pkl', 'rb') as f:
    gender_encoder = pickle.load(f)
with open('disease.pkl', 'rb') as f:
    disease_encoder = pickle.load(f)
with open('activity.pkl', 'rb') as f:
    activity_encoder = pickle.load(f)
with open('severity.pkl', 'rb') as f:
    severity_encoder = pickle.load(f)

# Get encoder classes
gender_classes = gender_encoder.classes_
disease_classes = disease_encoder.classes_
activity_classes = activity_encoder.classes_
severity_classes = severity_encoder.classes_

def reset_session():
    session.clear()
    session['stage'] = 'age'  # Start directly with age collection
    session['data'] = {
        'age': None,
        'weight': None,
        'height': None,
        'gender': None,
        'activity_level': None,
        'disease_type': None,
        'severity': None,
        'diet_preference': None
    }

def predict_recommended_calories(age, weight, height, gender, activity_level, severity, disease_type):
    try:
        if gender not in gender_classes:
            raise ValueError(f"Invalid gender. Must be one of: {', '.join(gender_classes)}")
        if disease_type not in disease_classes:
            raise ValueError(f"Invalid disease. Must be one of: {', '.join(disease_classes)}")
        if activity_level not in activity_classes:
            raise ValueError(f"Invalid activity level. Must be one of: {', '.join(activity_classes)}")
        
        gender_encoded = gender_encoder.transform([gender])[0]
        disease_encoded = disease_encoder.transform([disease_type])[0]
        activity_encoded = activity_encoder.transform([activity_level])[0]
        
        if disease_type == 'None':
            severity_encoded = severity_encoder.transform(['Mild'])[0]
        else:
            if severity not in severity_classes:
                raise ValueError(f"Invalid severity. Must be one of: {', '.join(severity_classes)}")
            severity_encoded = severity_encoder.transform([severity])[0]
        
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender_encoded],
            'Weight_kg': [weight],
            'Height_cm': [height],
            'Disease_Type': [disease_encoded],
            'Severity': [severity_encoded],
            'Physical_Activity_Level': [activity_encoded]
        })
        
        return round(model.predict(input_data)[0])
    except Exception as e:
        raise ValueError(f"Prediction error: {str(e)}")

def extract_calories(meal_string):
    match = re.search(r'\((\d+)\s*cal\)', meal_string)
    return int(match.group(1)) if match else 0

def find_best_meal_combination(meals_by_type, target_calories, max_combinations=1000):
    best_combination = None
    min_diff = float('inf')
    
    for _ in range(max_combinations):
        combination = {
            'breakfast': random.choice(meals_by_type['breakfast']),
            'lunch': random.choice(meals_by_type['lunch']),
            'dinner': random.choice(meals_by_type['dinner']),
            'snacks': random.choice(meals_by_type['snacks'])
        }
        
        total_calories = sum(extract_calories(meal) for meal in combination.values())
        diff = abs(total_calories - target_calories)
        
        if diff < min_diff:
            min_diff = diff
            best_combination = combination
            if diff == 0:
                break
    
    return best_combination, sum(extract_calories(meal) for meal in best_combination.values())

def recommend_meals(recommended_calories, disease_type, diet_preference):
    # Extensive Indian meal database
    meal_db = {
        'diabetes': {
            'veg': {
                'breakfast': [
                    "Masala oats with vegetables (280 cal)",
                    "Besan chilla with mint chutney (300 cal)"
                ],
                'lunch': [
                    "Palak paneer with 1 roti (400 cal)",
                    "Vegetable khichdi with curd (380 cal)"
                ],
                'dinner': [
                    "Vegetable pulao with raita (350 cal)",
                    "Dudhi kofta curry with 1 roti (320 cal)"
                ],
                'snacks': [
                    "Roasted chana with lemon (150 cal)",
                    "Sprouts salad with cucumber (120 cal)"
                ]
            },
            'non-veg': {
                'breakfast': [
                    "Egg white bhurji with multigrain toast (280 cal)",
                    "Chicken sandwich with whole wheat bread (300 cal)"
                ],
                'lunch': [
                    "Grilled chicken with quinoa and vegetables (400 cal)",
                    "Fish curry with 1 roti (380 cal)"
                ],
                'dinner': [
                    "Baked fish with steamed vegetables (350 cal)",
                    "Chicken tikka with salad (320 cal)"
                ],
                'snacks': [
                    "Boiled eggs with pepper (150 cal)",
                    "Greek yogurt with berries (120 cal)"
                ]
            }
        },
        'none': {
            'veg': {
                'breakfast': [
                    "Poha with peanuts and vegetables (300 cal)",
                    "Upma with vegetables and coconut (320 cal)"
                ],
                'lunch': [
                    "Dal tadka with jeera rice and roti (450 cal)",
                    "Paneer butter masala with naan (500 cal)"
                ],
                'dinner': [
                    "Dal makhani with roti and salad (400 cal)",
                    "Vegetable pulao with raita (420 cal)"
                ],
                'snacks': [
                    "Fruit chaat with chaat masala (150 cal)",
                    "Roasted makhana with ghee (120 cal)"
                ]
            },
            'non-veg': {
                'breakfast': [
                    "Egg bhurji with toast and tea (350 cal)",
                    "Chicken sandwich with mayo (380 cal)"
                ],
                'lunch': [
                    "Butter chicken with naan and salad (550 cal)",
                    "Fish curry with steamed rice (500 cal)"
                ],
                'dinner': [
                    "Grilled chicken with mashed potatoes (450 cal)",
                    "Fish tikka with naan and salad (470 cal)"
                ],
                'snacks': [
                    "Boiled eggs with salt and pepper (150 cal)",
                    "Chicken soup with croutons (180 cal)"
                ]
            }
        }
    }

    disease_key = disease_type.lower() if disease_type.lower() in ['diabetes', 'hypertension', 'obesity'] else 'none'
    diet_key = 'veg' if diet_preference.lower() in ['vegetarian', 'veg'] else 'non-veg'
    
    best_combo, total_cals = find_best_meal_combination(
        meal_db[disease_key][diet_key], 
        recommended_calories
    )
    
    return {
        'meals': best_combo,
        'total_calories': total_cals,
        'recommended_calories': recommended_calories
    }

def format_meal_plan(meal_plan):
    formatted = "🍽 <b>YOUR PERSONALIZED MEAL PLAN</b> 🍽\n\n"
    meals = [
        ("🌅 BREAKFAST", meal_plan['meals']['breakfast']),
        ("🍲 LUNCH", meal_plan['meals']['lunch']), 
        ("🌙 DINNER", meal_plan['meals']['dinner']),
        ("🍎 SNACKS", meal_plan['meals']['snacks'])
    ]
    
    for meal_name, meal in meals:
        calories = extract_calories(meal)
        meal_desc = meal.split('(')[0].strip()
        formatted += f"<b>{meal_name}</b> ({calories} calories)\n"
        formatted += f"→ {meal_desc}\n\n"
    
    formatted += "Would you like to start over? (yes/no)"
    return formatted

def chatbot_response(user_input):
    user_input = user_input.lower().strip()
    current_stage = session.get('stage', 'age')
    data = session.get('data', {})

    if current_stage == 'age':
        try:
            age = int(user_input)
            if not 1 <= age <= 120:
                return "Please enter a valid age between 1-120 years."
            data['age'] = age
            session['data'] = data
            session['stage'] = 'weight'
            return "Thank you! Now, please tell me your weight in kilograms?"
        except ValueError:
            return "Please enter a valid number for age."

    elif current_stage == 'weight':
        try:
            weight = float(user_input)
            if not 10 <= weight <= 300:
                return "Please enter weight between 10-300 kg."
            data['weight'] = weight
            session['data'] = data
            session['stage'] = 'height'
            return "Thank you! Now please tell me your height in centimeters?"
        except ValueError:
            return "Please enter a valid number for weight."

    elif current_stage == 'height':
        try:
            height = float(user_input)
            if not 50 <= height <= 250:
                return "Please enter height between 50-250 cm."
            data['height'] = height
            session['data'] = data
            session['stage'] = 'gender'
            return f"What is your gender? ({', '.join(gender_classes)})"
        except ValueError:
            return "Please enter a valid number for height."

    elif current_stage == 'gender':
        gender = user_input.capitalize()
        if gender not in gender_classes:
            return f"Please choose from: {', '.join(gender_classes)}"
        data['gender'] = gender
        session['data'] = data
        session['stage'] = 'activity'
        return f"What is your physical activity level? ({', '.join(activity_classes)})"

    elif current_stage == 'activity':
        activity = user_input.capitalize()
        if activity not in activity_classes:
            return f"Please choose from: {', '.join(activity_classes)}"
        data['activity_level'] = activity
        session['data'] = data
        session['stage'] = 'disease'
        return f"Do you have any health conditions? ({', '.join(disease_classes)})"

    elif current_stage == 'disease':
        disease = user_input.capitalize()
        if disease not in disease_classes:
            return f"Please choose from: {', '.join(disease_classes)}"
        data['disease_type'] = disease
        session['data'] = data
        if disease == 'None':
            data['severity'] = 'Mild'
            session['stage'] = 'diet_preference'
            return "Do you prefer vegetarian or non-vegetarian meals? (veg/non-veg)"
        else:
            session['stage'] = 'severity'
            return f"How severe is your {disease}? ({', '.join(s for s in severity_classes if s != 'none')})"

    elif current_stage == 'severity':
        severity = user_input.capitalize()
        if severity not in severity_classes or severity == 'none':
            return f"Please choose from: {', '.join(s for s in severity_classes if s != 'none')}"
        data['severity'] = severity
        session['data'] = data
        session['stage'] = 'diet_preference'
        return "Do you prefer vegetarian or non-vegetarian meals? (veg/non-veg)"

    elif current_stage == 'diet_preference':
        diet = user_input.lower()
        if diet not in ['vegetarian', 'non-vegetarian', 'veg', 'non-veg']:
            return "Please specify 'veg' or 'non-veg'"
        data['diet_preference'] = 'veg' if diet in ['vegetarian', 'veg'] else 'non-veg'
        session['data'] = data
        session['stage'] = 'complete'
        
        try:
            params = {
                'age': data['age'],
                'weight': data['weight'],
                'height': data['height'],
                'gender': data['gender'],
                'activity_level': data['activity_level'],
                'severity': data.get('severity', 'Mild'),
                'disease_type': data['disease_type']
            }
            
            recommended_calories = predict_recommended_calories(**params)
            meal_plan = recommend_meals(recommended_calories, data['disease_type'], data['diet_preference'])
            
            response = "🔹 <b>Your Personalized Diet Plan</b> 🔹\n\n"
            response += f"<b>Recommended Daily Calories:</b> {recommended_calories}\n\n"
            response += format_meal_plan(meal_plan)
            return response
        except Exception as e:
            return f"Sorry, there was an error: {str(e)}"

    elif current_stage == 'complete':
        if user_input.lower() in ['yes', 'y']:
            reset_session()
            return "Great! Let's start again. What is your age?"
        else:
            return "Thank you for using our Diet Assistant! Have a healthy day!"

    else:
        reset_session()
        return "Let's start over. What is your age?"

@app.before_request
def before_request():
    session.permanent = True
    if 'stage' not in session:
        reset_session()
    # Ensure all required session keys exist
    if 'data' not in session:
        session['data'] = {
            'age': None,
            'weight': None,
            'height': None,
            'gender': None,
            'activity_level': None,
            'disease_type': None,
            'severity': None,
            'diet_preference': None
        }

@app.route('/')
def home():
    return render_template('chatbot.html')

@app.route('/start', methods=['POST'])
def start_conversation():
    reset_session()
    return jsonify({
        'response': "Welcome to Diet Assistant! What is your age?",
        'status': 'success'
    })

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Ensure session exists
        if 'stage' not in session:
            reset_session()
            
        user_input = request.json.get('message', '').strip()
        response = chatbot_response(user_input)
        
        # Explicitly mark session as modified
        session.modified = True
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        reset_session()
        return jsonify({
            'response': "Sorry, there was an error. Let's start over. What is your age?",
            'status': 'error'
        })

if __name__ == '__main__':
    app.run(debug=True)