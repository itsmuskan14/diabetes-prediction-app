import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import requests
import json

# ------------------ SETUP ------------------
st.set_page_config(page_title="Diabetes Prediction System", layout="centered")

# Initialize page state
if "page" not in st.session_state:
    st.session_state.page = "gender"

# ------------------ LOAD & TRAIN MODEL ------------------
@st.cache_resource
def load_and_train_model():
    """Load data and train model (cached for performance)"""
    try:
        data = pd.read_csv("diabetes.csv")
    except FileNotFoundError:
        st.error("⚠️ diabetes.csv file not found. Please ensure the file is in the same directory as app.py")
        st.stop()
    
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    
    return model, data

@st.cache_data
def load_doctors_database():
    """Load doctors database (cached for performance)"""
    try:
        doctors_db = pd.read_csv("doctors_database.csv")
        return doctors_db
    except FileNotFoundError:
        st.warning("⚠️ doctors_database.csv not found. Will use API search only.")
        return pd.DataFrame()

model, data = load_and_train_model()
doctors_db = load_doctors_database()

# ------------------ LOCATION & DOCTOR FINDER FUNCTIONS ------------------
def get_coordinates_from_location(location_query):
    """Get coordinates from location using Nominatim (OpenStreetMap)"""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': location_query,
            'format': 'json',
            'limit': 1
        }
        headers = {
            'User-Agent': 'DiabetesPredictionApp/1.0'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200 and response.json():
            data = response.json()[0]
            return {
                'lat': float(data['lat']),
                'lon': float(data['lon']),
                'display_name': data['display_name']
            }
        return None
    except Exception as e:
        st.error(f"Error getting location: {str(e)}")
        return None

def find_nearby_doctors(lat, lon, city=None, radius=5000):
    """Find nearby doctors using local database first, then API as backup"""
    facilities = []
    
    # PRIORITY 1: Search local database first
    if not doctors_db.empty:
        # If city is provided, filter by city first
        if city:
            # Try exact city match
            city_matches = doctors_db[doctors_db['city'].str.lower() == city.lower()]
            
            # If no exact match, try partial match
            if city_matches.empty:
                city_matches = doctors_db[doctors_db['city'].str.contains(city, case=False, na=False)]
            
            # Calculate distances for city matches
            for _, row in city_matches.iterrows():
                distance = calculate_distance(lat, lon, row['lat'], row['lon'])
                facilities.append({
                    'name': row['name'],
                    'type': row['type'],
                    'specialty': row['specialty'],
                    'address': row['address'],
                    'phone': row['phone'],
                    'website': row['website'],
                    'lat': row['lat'],
                    'lon': row['lon'],
                    'distance': distance,
                    'rating': row.get('rating', 'N/A'),
                    'source': 'database'
                })
        
        # Also search nearby locations from database
        for _, row in doctors_db.iterrows():
            distance = calculate_distance(lat, lon, row['lat'], row['lon'])
            if distance <= radius / 1000:  # Convert radius to km
                # Check if already added
                if not any(f['name'] == row['name'] and f['address'] == row['address'] for f in facilities):
                    facilities.append({
                        'name': row['name'],
                        'type': row['type'],
                        'specialty': row['specialty'],
                        'address': row['address'],
                        'phone': row['phone'],
                        'website': row['website'],
                        'lat': row['lat'],
                        'lon': row['lon'],
                        'distance': distance,
                        'rating': row.get('rating', 'N/A'),
                        'source': 'database'
                    })
    
    # PRIORITY 2: If database has fewer than 5 results, supplement with API
    if len(facilities) < 5:
        try:
            overpass_url = "https://overpass-api.de/api/interpreter"
            
            query = f"""
            [out:json][timeout:25];
            (
              node["amenity"="hospital"](around:{radius},{lat},{lon});
              node["amenity"="clinic"](around:{radius},{lat},{lon});
              way["amenity"="hospital"](around:{radius},{lat},{lon});
              way["amenity"="clinic"](around:{radius},{lat},{lon});
            );
            out center;
            """
            
            response = requests.get(overpass_url, params={'data': query}, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                for element in data.get('elements', []):
                    if 'tags' in element:
                        tags = element['tags']
                        
                        if 'lat' in element and 'lon' in element:
                            elem_lat = element['lat']
                            elem_lon = element['lon']
                        elif 'center' in element:
                            elem_lat = element['center']['lat']
                            elem_lon = element['center']['lon']
                        else:
                            continue
                        
                        name = tags.get('name', tags.get('operator', 'Healthcare Facility'))
                        address = format_address(tags)
                        
                        # Check if already in facilities
                        if not any(f['name'] == name and f['address'] == address for f in facilities):
                            distance = calculate_distance(lat, lon, elem_lat, elem_lon)
                            facilities.append({
                                'name': name,
                                'type': tags.get('amenity', 'healthcare').title(),
                                'specialty': tags.get('healthcare:speciality', 'General Medicine'),
                                'address': address,
                                'phone': tags.get('phone', tags.get('contact:phone', 'N/A')),
                                'website': tags.get('website', tags.get('contact:website', 'N/A')),
                                'lat': elem_lat,
                                'lon': elem_lon,
                                'distance': distance,
                                'rating': 'N/A',
                                'source': 'api'
                            })
        
        except Exception as e:
            st.warning(f"⚠️ API search unavailable: {str(e)}")
    
    # Sort by distance and return top results
    facilities.sort(key=lambda x: x.get('distance', float('inf')))
    return facilities[:15] if facilities else []

def format_address(tags):
    """Format address from OSM tags"""
    address_parts = []
    
    if tags.get('addr:housenumber'):
        address_parts.append(tags['addr:housenumber'])
    if tags.get('addr:street'):
        address_parts.append(tags['addr:street'])
    if tags.get('addr:suburb'):
        address_parts.append(tags['addr:suburb'])
    if tags.get('addr:city'):
        address_parts.append(tags['addr:city'])
    if tags.get('addr:state'):
        address_parts.append(tags['addr:state'])
    
    return ', '.join(address_parts) if address_parts else 'Address not available'

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate approximate distance in km using Haversine formula"""
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371  # Earth's radius in km
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return round(distance, 2)

def get_google_maps_link(lat, lon, facility_name):
    """Generate Google Maps link"""
    query = facility_name.replace(' ', '+')
    return f"https://www.google.com/maps/search/?api=1&query={lat},{lon}&query={query}"

# ------------------ DIET & MEDICATION DATA ------------------
diet_plan = {
    "Breakfast": [
        "Oatmeal with berries and nuts",
        "Whole grain toast with avocado",
        "Greek yogurt with flaxseeds",
        "Vegetable omelet with whole wheat toast"
    ],
    "Lunch": [
        "Grilled chicken or fish with quinoa and vegetables",
        "Lentil soup with mixed green salad",
        "Brown rice with dal and vegetables",
        "Whole wheat roti with vegetables and lean protein"
    ],
    "Dinner": [
        "Baked salmon with steamed broccoli",
        "Vegetable stir-fry with tofu",
        "Grilled chicken with roasted vegetables",
        "Vegetable soup with whole grain bread"
    ],
    "Snacks": [
        "Apple slices with almond butter",
        "Carrot and cucumber sticks with hummus",
        "Handful of mixed nuts (unsalted)",
        "Low-fat yogurt"
    ],
    "Foods to Avoid": [
        "❌ Sugary drinks (soda, sweetened juices)",
        "❌ White bread, white rice, refined pasta",
        "❌ Pastries, cakes, cookies",
        "❌ Fried foods and fast food",
        "❌ Processed meats",
        "❌ High-sugar cereals"
    ]
}

general_tips = [
    "🥤 Drink at least 8 glasses of water daily",
    "🏃 Exercise for 30 minutes, 5 days a week",
    "😴 Get 7-8 hours of quality sleep",
    "📊 Monitor blood sugar levels regularly",
    "🧘 Practice stress management techniques",
    "🚭 Avoid smoking and limit alcohol"
]

medications = [
    "**Metformin** - Most commonly prescribed first-line medication",
    "**Sulfonylureas** - Helps pancreas release more insulin",
    "**DPP-4 inhibitors** - Helps reduce blood sugar levels",
    "**GLP-1 receptor agonists** - Injectable medication for Type 2 diabetes",
    "**SGLT2 inhibitors** - Helps kidneys remove sugar through urine",
    "**Insulin therapy** - May be required in advanced cases"
]

# ------------------ PAGE 1: GENDER ------------------
if st.session_state.page == "gender":
    st.title("🩺 Diabetes Prediction System")
    st.markdown("### Welcome to the AI-powered Diabetes Risk Assessment")
    st.info("ℹ️ This tool uses machine learning to predict diabetes risk based on medical indicators.")
    
    st.markdown("---")
    st.subheader("Step 1: Select Your Gender")

    gender = st.radio("Gender", ("Female", "Male"), horizontal=True, key="gender_select")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("➡️ Continue to Assessment", type="primary", use_container_width=True):
            st.session_state.gender = gender
            st.session_state.page = "location"
            st.rerun()

# ------------------ PAGE 2: LOCATION ------------------
elif st.session_state.page == "location":
    st.title("📍 Your Location")
    st.write(f"**Gender:** {st.session_state.gender}")
    
    st.markdown("---")
    st.subheader("Step 2: Enter Your Location")
    st.info("💡 This helps us find nearby doctors and healthcare facilities if needed.")
    
    # Location input options
    location_method = st.radio(
        "Choose how to provide your location:",
        ("Enter City/Address", "Skip (I'll find doctors myself)"),
        key="location_method"
    )
    
    if location_method == "Enter City/Address":
        location_input = st.text_input(
            "Enter your city, area, or full address:",
            placeholder="e.g., Mumbai, Maharashtra or New York, NY or London, UK",
            help="Be as specific as possible for better results"
        )
        
        if location_input:
            if st.button("🔍 Verify Location", type="secondary"):
                with st.spinner("Verifying location..."):
                    coords = get_coordinates_from_location(location_input)
                    if coords:
                        st.success(f"✅ Location found: {coords['display_name']}")
                        st.session_state.location = location_input
                        st.session_state.coordinates = coords
                        st.session_state.location_verified = True
                    else:
                        st.error("❌ Could not find location. Please try again with a different format.")
                        st.session_state.location_verified = False
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Back", type="secondary"):
            st.session_state.page = "gender"
            st.rerun()
    
    with col3:
        # Allow proceeding if location is verified or skipped
        can_proceed = (
            location_method == "Skip (I'll find doctors myself)" or 
            st.session_state.get('location_verified', False)
        )
        
        if can_proceed:
            if st.button("➡️ Continue", type="primary"):
                if location_method == "Skip (I'll find doctors myself)":
                    st.session_state.location = None
                    st.session_state.coordinates = None
                st.session_state.page = "predict"
                st.rerun()
        else:
            st.button("➡️ Continue", type="primary", disabled=True, 
                     help="Please verify your location first or choose to skip")

# ------------------ PAGE 3: PREDICTION FORM ------------------
elif st.session_state.page == "predict":
    st.title("🧪 Medical Assessment & Prediction")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**Gender:** {st.session_state.gender}")
        if st.session_state.get('location'):
            st.write(f"**Location:** {st.session_state.location}")
    with col2:
        if st.button("⬅️ Change", type="secondary"):
            st.session_state.page = "gender"
            st.rerun()
    
    st.markdown("---")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.gender == "Female":
            preg = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1, 
                                   help="Total number of times pregnant")
        else:
            st.info("👨 Pregnancy: Not applicable (set to 0)")
            preg = 0
        
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=50, max_value=300, value=120,
                                  help="Normal fasting: 70-100 mg/dL | Post-meal: < 140 mg/dL")
        
        bp = st.number_input("Blood Pressure (mm Hg)", min_value=40, max_value=200, value=80,
                            help="Diastolic blood pressure. Normal: 60-80 mm Hg")
        
        skin = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20,
                              help="Triceps skin fold thickness. Normal: 10-40 mm")
    
    with col2:
        insulin = st.number_input("Insulin Level (µU/mL)", min_value=0, max_value=900, value=80,
                                 help="2-hour serum insulin. Normal: 16-166 µU/mL")
        
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=70.0, value=25.0, step=0.1,
                             help="Weight(kg) / Height(m)². Normal: 18.5-24.9")
        
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01,
                             help="Genetic predisposition to diabetes. Typical: 0.1-1.0")
        
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)

    # Medical History
    st.markdown("---")
    st.subheader("🩺 Medical History (Optional)")
    
    col1, col2 = st.columns(2)
    with col1:
        hypertension = st.checkbox("High Blood Pressure")
        family_diabetes = st.checkbox("Family History of Diabetes")
    with col2:
        smoking = st.checkbox("Smoker")
        physical_activity = st.selectbox("Physical Activity Level", 
                                         ["Sedentary", "Light", "Moderate", "Active"])
    
    other_conditions = st.text_area("Other Medical Conditions (if any)", 
                                    placeholder="e.g., heart disease, thyroid issues, PCOS...")

    # Predict Button
    st.markdown("---")
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    with predict_col2:
        predict_button = st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True)
    
    if predict_button:
        input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        
        result = model.predict(input_data)
        prob = model.predict_proba(input_data)[0]
        diabetic_prob = prob[1] * 100
        non_diabetic_prob = prob[0] * 100
        
        st.markdown("---")
        st.markdown("## 📊 Prediction Results")
        
        if result[0] == 1:
            st.error(f"### ⚠️ HIGH RISK - Diabetes Detected")
            st.metric("Probability of Diabetes", f"{diabetic_prob:.1f}%", 
                     delta=None, delta_color="normal")
            
            # Risk factors analysis
            st.markdown("---")
            st.subheader("🔍 Risk Factor Analysis")
            
            risk_factors = []
            if glucose > 140:
                risk_factors.append("🔴 High glucose level detected")
            if bmi > 30:
                risk_factors.append("🔴 BMI indicates obesity")
            if age > 45:
                risk_factors.append("🟡 Age is a risk factor")
            if bp > 90:
                risk_factors.append("🔴 Elevated blood pressure")
            if insulin > 200:
                risk_factors.append("🔴 High insulin levels")
            if family_diabetes:
                risk_factors.append("🟡 Family history present")
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(factor)
            
            # DOCTOR FINDER SECTION
            st.markdown("---")
            st.markdown("## 🏥 Find Nearby Healthcare Facilities")
            
            if st.session_state.get('coordinates'):
                with st.spinner("🔍 Searching for nearby doctors and hospitals..."):
                    coords = st.session_state.coordinates
                    city = st.session_state.get('location', '').split(',')[0].strip()
                    facilities = find_nearby_doctors(coords['lat'], coords['lon'], city=city, radius=50000)
                    
                    if facilities:
                        st.success(f"✅ Found {len(facilities)} healthcare facilities near you!")
                        
                        # Show summary stats
                        db_count = sum(1 for f in facilities if f.get('source') == 'database')
                        if db_count > 0:
                            st.info(f"📚 {db_count} verified facilities from our database | {len(facilities) - db_count} from online sources")
                        
                        # Display facilities
                        for idx, facility in enumerate(facilities, 1):
                            # Icon based on type and source
                            icon = '🏥' if facility['type'] == 'Hospital' else '🏨'
                            verified = '✅' if facility.get('source') == 'database' else ''
                            
                            with st.expander(f"{icon} {verified} {idx}. {facility['name']} - {facility['distance']:.1f} km away"):
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.write(f"**Type:** {facility['type']}")
                                    st.write(f"**Specialty:** {facility['specialty']}")
                                    
                                    if facility.get('rating') and facility['rating'] != 'N/A':
                                        st.write(f"**Rating:** ⭐ {facility['rating']}/5.0")
                                    
                                    if facility['address'] and facility['address'] != 'Address not available':
                                        st.write(f"**Address:** {facility['address']}")
                                    
                                    if facility['phone'] != 'N/A':
                                        st.write(f"**Phone:** {facility['phone']}")
                                    
                                    st.write(f"**Distance:** {facility['distance']:.1f} km")
                                
                                with col2:
                                    maps_link = get_google_maps_link(facility['lat'], facility['lon'], facility['name'])
                                    st.markdown(f"[📍 Directions]({maps_link})")
                                    
                                    if facility['website'] and facility['website'] != 'N/A':
                                        st.markdown(f"[🌐 Website]({facility['website']})")
                                    
                                    # Call button
                                    if facility['phone'] != 'N/A':
                                        st.markdown(f"[📞 Call]({facility['phone'].replace(' ', '')})")
                        
                        st.markdown("---")
                        st.info("💡 **Tip:** ✅ indicates verified facilities from our curated database. Click '📍 Directions' for Google Maps navigation.")
                    
                    else:
                        st.warning("⚠️ Could not find healthcare facilities in our database for this area.")
                        st.markdown("### 🔍 Alternative Ways to Find Doctors:")
                        
                        location_query = st.session_state.get('location', 'your area')
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 🌐 Search Online:")
                            google_search = f"https://www.google.com/maps/search/endocrinologist+near+{location_query.replace(' ', '+')}"
                            st.markdown(f"[🔍 Google Maps: Endocrinologists]({google_search})")
                            
                            google_hospital = f"https://www.google.com/maps/search/hospital+near+{location_query.replace(' ', '+')}"
                            st.markdown(f"[🏥 Google Maps: Hospitals]({google_hospital})")
                            
                            google_diabetes = f"https://www.google.com/search?q=diabetes+specialist+in+{location_query.replace(' ', '+')}"
                            st.markdown(f"[🔬 Google: Diabetes Specialists]({google_diabetes})")
                        
                        with col2:
                            st.markdown("#### 📱 Healthcare Apps:")
                            st.markdown("[🩺 Practo (India)](https://www.practo.com)")
                            st.markdown("[🏥 Zocdoc (USA)](https://www.zocdoc.com)")
                            st.markdown("[💊 Healthgrades](https://www.healthgrades.com)")
                            st.markdown("[🌍 WebMD Physician Directory](https://doctor.webmd.com)")
                        
                        st.markdown("---")
                        st.info("""
                        **💡 Tips for Finding the Right Doctor:**
                        - Look for **Endocrinologists** (diabetes specialists)
                        - Check **reviews and ratings** from other patients
                        - Verify they accept your **insurance** (if applicable)
                        - Call ahead to confirm they're accepting **new patients**
                        - Ask about **telemedicine** options if available
                        """)
            
            else:
                st.info("📍 **Location not provided.** Here are some general recommendations:")
                
                # Show some top hospitals from database
                if not doctors_db.empty:
                    st.markdown("### 🏥 Top Recommended Hospitals:")
                    top_hospitals = doctors_db.nlargest(5, 'rating')
                    
                    for _, hospital in top_hospitals.iterrows():
                        with st.expander(f"⭐ {hospital['name']} - {hospital['city']}, {hospital['country']}"):
                            st.write(f"**Rating:** ⭐ {hospital['rating']}/5.0")
                            st.write(f"**Type:** {hospital['type']}")
                            st.write(f"**Specialty:** {hospital['specialty']}")
                            st.write(f"**Address:** {hospital['address']}")
                            if hospital['phone'] != 'N/A':
                                st.write(f"**Phone:** {hospital['phone']}")
                            if hospital['website'] != 'N/A':
                                st.markdown(f"[🌐 Visit Website]({hospital['website']})")
                    
                    st.markdown("---")
                
                st.write("**How to find a diabetes specialist:**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🔍 Quick Search Options:")
                    st.markdown("[🗺️ Google Maps - Search Nearby](https://www.google.com/maps/search/endocrinologist)")
                    st.markdown("[🌐 General Google Search](https://www.google.com/search?q=diabetes+specialist+near+me)")
                
                with col2:
                    st.markdown("#### 📱 Popular Healthcare Apps:")
                    st.markdown("[🩺 Practo](https://www.practo.com)")
                    st.markdown("[🏥 Zocdoc](https://www.zocdoc.com)")
                
                st.markdown("---")
                st.write("**Other Options:**")
                st.write("1. 🏥 Visit your nearest hospital's endocrinology department")
                st.write("2. 📞 Contact your insurance provider for in-network specialists")
                st.write("3. 💬 Ask your primary care physician for a referral")
                st.write("4. 🔎 Search for 'endocrinologist' or 'diabetologist' in your area")
            
            # Diet Plan
            st.markdown("---")
            st.subheader("🍎 Recommended Diet Plan")
            st.info("⚕️ Please consult with a registered dietitian for a personalized meal plan.")
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌅 Breakfast", "🍽️ Lunch", "🌙 Dinner", "🥜 Snacks", "❌ Avoid"])
            
            with tab1:
                for item in diet_plan["Breakfast"]:
                    st.write(f"• {item}")
            
            with tab2:
                for item in diet_plan["Lunch"]:
                    st.write(f"• {item}")
            
            with tab3:
                for item in diet_plan["Dinner"]:
                    st.write(f"• {item}")
            
            with tab4:
                for item in diet_plan["Snacks"]:
                    st.write(f"• {item}")
            
            with tab5:
                for item in diet_plan["Foods to Avoid"]:
                    st.write(item)
            
            # General Tips
            st.markdown("---")
            st.subheader("💡 Lifestyle Recommendations")
            for tip in general_tips:
                st.write(tip)
            
            # Medications
            st.markdown("---")
            st.subheader("💊 Common Diabetes Medications")
            st.warning("⚠️ **IMPORTANT:** Consult your doctor before starting any medication. This is for informational purposes only.")
            
            for med in medications:
                st.write(f"• {med}")
            
            # Next Steps
            st.markdown("---")
            st.subheader("🏥 Recommended Next Steps")
            st.write("1. **Schedule an appointment** with your healthcare provider immediately")
            st.write("2. **Get comprehensive blood tests** including HbA1c, fasting glucose")
            st.write("3. **Start monitoring** your blood sugar levels daily")
            st.write("4. **Begin lifestyle modifications** as recommended above")
            st.write("5. **Consider consulting** an endocrinologist for specialized care")
            
        else:
            st.success(f"### ✅ LOW RISK - No Diabetes Detected")
            st.metric("Probability of No Diabetes", f"{non_diabetic_prob:.1f}%", 
                     delta=None, delta_color="normal")
            
            st.info("**Good news!** Based on the provided information, you have a low risk of diabetes.")
            
            # Prevention tips
            st.markdown("---")
            st.subheader("🛡️ Prevention Tips")
            st.write("Even with low risk, maintaining a healthy lifestyle is important:")
            
            prevention_tips = [
                "✅ Maintain a healthy weight (BMI 18.5-24.9)",
                "✅ Exercise regularly (at least 150 minutes per week)",
                "✅ Eat a balanced diet rich in vegetables and whole grains",
                "✅ Limit sugar and processed foods",
                "✅ Get regular health checkups (annually)",
                "✅ Monitor blood pressure and cholesterol levels",
                "✅ Manage stress through meditation or yoga",
                "✅ Get adequate sleep (7-8 hours per night)"
            ]
            
            for tip in prevention_tips:
                st.write(tip)
            
            st.markdown("---")
            st.info("💡 **Recommendation:** Continue with annual health screenings and maintain your healthy lifestyle!")
        
        # Display input summary
        with st.expander("📋 View Input Summary"):
            summary_data = {
                "Parameter": ["Gender", "Location", "Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", 
                             "Insulin", "BMI", "Diabetes Pedigree", "Age"],
                "Value": [st.session_state.gender, 
                         st.session_state.get('location', 'Not provided'),
                         preg, glucose, bp, skin, insulin, f"{bmi:.1f}", 
                         f"{dpf:.3f}", age],
                "Unit": ["", "", "count", "mg/dL", "mm Hg", "mm", "µU/mL", "kg/m²", "score", "years"]
            }
            st.table(pd.DataFrame(summary_data))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice. 
    Always consult with healthcare professionals for diagnosis and treatment.</p>
    <p>Powered by Machine Learning | Location Services by OpenStreetMap | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)