import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import requests
import plotly.graph_objects as go


# ------------------ 1. CONFIG & STYLING ------------------
st.set_page_config(page_title="DiaCare-Predict", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

# -------------------- CUSTOM CSS : DIACARE HACKATHON UI --------------------
# -------------------- FINAL STABLE UI THEME --------------------
st.markdown("""
<style>

    /* --- 1. HEADER FIX --- */
    /* Makes the top header bar transparent to match the background color */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
  /* --- HEADING SIZE FIX --- */
    
    /* Force H1 (Titles) to be large */
    h1, .main h1 {
        font-size: 36px !important;
        font-weight: 700 !important;
    }
    
    /* Force H2 (Subheaders) to be medium-large */
    h2, .main h2 {
        font-size: 28px !important;
        font-weight: 600 !important;
        margin-top: 20px !important;
    }
    
    /* Force H3 (Section Titles) */
    h3, .main h3 {
        font-size: 22px !important;
        font-weight: 600 !important;
    }
    
    /* Force H4 (Small titles inside cards) */
    h4, .main h4 {
        font-size: 18px !important;
        font-weight: 600 !important;
    }
            
/* =====================================================
   1. FULL APP BACKGROUND (Dark Blue Gradient)
===================================================== */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

/* =====================================================
   2. MAIN CONTENT CARD (Right Panel)
===================================================== */
.block-container {
    max-width: 950px;
    margin-top: 40px;
    background-color: #f9fafb;   /* Soft light gray */
    padding: 40px 45px;
    border-radius: 26px;
    box-shadow: 0 30px 60px rgba(0,0,0,0.35);
}

/* =====================================================
   3. SIDEBAR (Left Panel)
===================================================== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #0f172a);
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

/* --- SIDEBAR RESET BUTTON (FORCE OVERRIDE) --- */
    
    /* Target the button specifically inside the sidebar */
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #e74c3c !important; /* Red Background */
        color: white !important;              /* White Text */
        border: 1px solid #c0392b !important; /* Darker Red Border */
        font-weight: bold !important;
    }
    
    /* Hover Effect */
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #c0392b !important; /* Darker Red on Hover */
        color: white !important;
        border-color: #a93226 !important;
    }
    
    /* Remove any white background on focus */
    section[data-testid="stSidebar"] .stButton > button:focus {
        background-color: #e74c3c !important;
        color: white !important;
        box-shadow: none !important;
    }
            
/* =====================================================
   4. TEXT FIX (THIS SOLVES YOUR PROBLEM)
===================================================== */
.block-container h1,
.block-container h2,
.block-container h3,
.block-container h4 {
    color: #020617 !important;
    font-weight: 600;
}

.block-container p,
.block-container span,
.block-container label {
    color: #1e293b !important;
    font-size: 15px;
}

/* Radio buttons */
div[role="radiogroup"] label {
    color: #1e293b !important;
    font-weight: 500;
}

/* =====================================================
   5. INFO / HELP BOX (Blue Card in Screenshot)
===================================================== */
.stInfo {
    background-color: #dbeafe !important;
    color: #1e40af !important;
    border-radius: 12px;
    font-size: 14px;
}

/* =====================================================
   6. INPUT FIELDS
===================================================== */
input, textarea, select {
    background-color: #ffffff !important;
    color: #020617 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
}

/* =====================================================
   7. BUTTON (Next Step)
===================================================== */
div.stButton > button {
    border-radius: 12px;
    padding: 10px 22px;
    font-weight: 600;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #22c55e, #16a34a);
    color: white !important;
    border: none;
}

div.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(34,197,94,0.45);
}

/* =====================================================
   8. REMOVE STREAMLIT TOP PADDING
===================================================== */
header {
    background: transparent;
}

            /* =====================================================
   FIX: SECONDARY BUTTON (Verify Location)
===================================================== */
div.stButton > button[kind="secondary"] {
    background-color: #e5e7eb !important;   /* Light gray */
    color: #020617 !important;              /* Dark text */
    border: 1px solid #cbd5e1 !important;
    font-weight: 600;
    border-radius: 12px;
}

/* Hover state */
div.stButton > button[kind="secondary"]:hover {
    background-color: #dbeafe !important;
    color: #1e40af !important;
}
            
/* =====================================================
   GLOBAL TEXT SAFETY FIX (Right Panel)
===================================================== */

/* Force readable dark text everywhere in main content */
section.main * {
    color: #0f172a !important;  /* Dark slate */
}

/* =====================================================
   SUCCESS / INFO / WARNING BOX FIX
===================================================== */

/* Success message (e.g., Location Verified) */
div[data-testid="stSuccess"] {
    background-color: #dcfce7 !important;
    border-left: 6px solid #16a34a;
}
div[data-testid="stSuccess"] * {
    color: #14532d !important;
    font-weight: 600;
}

/* Info messages */
div[data-testid="stInfo"] {
    background-color: #e0f2fe !important;
    border-left: 6px solid #0284c7;
}
div[data-testid="stInfo"] * {
    color: #0c4a6e !important;
}

/* Warning messages */
div[data-testid="stWarning"] {
    background-color: #fef3c7 !important;
}
div[data-testid="stWarning"] * {
    color: #78350f !important;
}

/* =====================================================
   PREDICTION PAGE RESULT TEXT FIX
===================================================== */

/* Markdown & result text */
div[data-testid="stMarkdown"] * {
    color: #020617 !important;
}

/* Metrics (Prediction values) */
div[data-testid="metric-container"] * {
    color: #020617 !important;
    font-weight: 600;
}

/* --- SIDEBAR (LEFT PANEL) FIX --- */
    
    /* 1. Set Background to Dark Navy */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
    }
    
    /* 2. Force ALL generic text to be White */
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* 3. Specific fix for "Step 1, Step 2" list items */
    /* This overrides the default Streamlit styling that was making them black/invisible */
    section[data-testid="stSidebar"] .stMarkdown p, 
    section[data-testid="stSidebar"] .stMarkdown li, 
    section[data-testid="stSidebar"] .stMarkdown h1, 
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #ffffff !important;
    }
            
</style>
""", unsafe_allow_html=True)




# Initialize page state
if "page" not in st.session_state:
    st.session_state.page = "gender"

# ------------------ 2. BACKEND LOGIC (PRESERVED) ------------------
@st.cache_resource
def load_and_train_model():
    """Load data and train model (cached for performance)"""
    try:
        data = pd.read_csv("diabetes.csv")
    except FileNotFoundError:
        # Fallback for UI demo if file missing - remove this in production if strict
        st.error("⚠️ diabetes.csv file not found. Please ensure the file is in the directory.")
        st.stop()
    
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model, data

@st.cache_data
def load_doctors_database():
    try:
        doctors_db = pd.read_csv("doctors_database.csv")
        return doctors_db
    except FileNotFoundError:
        return pd.DataFrame()

model, data = load_and_train_model()
doctors_db = load_doctors_database()

# --- Location Functions ---
def get_coordinates_from_location(location_query):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {'q': location_query, 'format': 'json', 'limit': 1}
        headers = {'User-Agent': 'DiabetesPredictionApp/1.0'}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200 and response.json():
            data = response.json()[0]
            return {'lat': float(data['lat']), 'lon': float(data['lon']), 'display_name': data['display_name']}
        return None
    except Exception as e:
        return None

def calculate_distance(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return round(R * c, 2)

def find_nearby_doctors(lat, lon, city=None, radius=5000):
    facilities = []
    # 1. Local DB Search
    if not doctors_db.empty:
        if city:
            city_matches = doctors_db[doctors_db['city'].str.contains(city, case=False, na=False)]
            for _, row in city_matches.iterrows():
                dist = calculate_distance(lat, lon, row['lat'], row['lon'])
                facilities.append({**row.to_dict(), 'distance': dist, 'source': 'database'})
        
        for _, row in doctors_db.iterrows():
            dist = calculate_distance(lat, lon, row['lat'], row['lon'])
            if dist <= radius / 1000:
                if not any(f['name'] == row['name'] for f in facilities):
                    facilities.append({**row.to_dict(), 'distance': dist, 'source': 'database'})

    # 2. API Backup
    if len(facilities) < 5:
        try:
            overpass_url = "https://overpass-api.de/api/interpreter"
            query = f"""[out:json][timeout:25];
            (node["amenity"="hospital"](around:{radius},{lat},{lon});
             node["amenity"="clinic"](around:{radius},{lat},{lon}););
            out center;"""
            response = requests.get(overpass_url, params={'data': query}, timeout=30)
            if response.status_code == 200:
                for element in response.json().get('elements', []):
                    tags = element.get('tags', {})
                    lat_e = element.get('lat', element.get('center', {}).get('lat'))
                    lon_e = element.get('lon', element.get('center', {}).get('lon'))
                    if lat_e and lon_e:
                        name = tags.get('name', 'Healthcare Facility')
                        if not any(f['name'] == name for f in facilities):
                            dist = calculate_distance(lat, lon, lat_e, lon_e)
                            facilities.append({
                                'name': name, 'type': tags.get('amenity', 'Hospital').title(),
                                'specialty': tags.get('healthcare:speciality', 'General'),
                                'address': tags.get('addr:street', 'Address unavailable'),
                                'phone': tags.get('phone', 'N/A'), 'website': tags.get('website', 'N/A'),
                                'lat': lat_e, 'lon': lon_e, 'distance': dist, 'rating': 'N/A', 'source': 'api'
                            })
        except: pass
    
    facilities.sort(key=lambda x: x.get('distance', float('inf')))
    return facilities[:15]

# --- Static Data ---
diet_plan = {
    "Breakfast": ["Oatmeal with berries", "Whole grain toast + avocado", "Greek yogurt + flaxseeds"],
    "Lunch": ["Grilled chicken + quinoa", "Lentil soup + salad", "Brown rice + dal"],
    "Dinner": ["Baked salmon + broccoli", "Tofu stir-fry", "Grilled chicken + veggies"],
    "Snacks": ["Apple + almond butter", "Carrot sticks + hummus", "Mixed nuts"],
    "Foods to Avoid": ["❌ Sugary drinks", "❌ White bread/rice", "❌ Fried foods", "❌ Processed meats"]
}
general_tips = ["🥤 Drink 8 glasses of water", "🏃 Exercise 30 mins/day", "😴 Sleep 7-8 hours", "🚭 Avoid smoking"]
medications = ["**Metformin**", "**Sulfonylureas**", "**DPP-4 inhibitors**", "**Insulin therapy**"]

# ------------------ 3. UI HELPERS ------------------
def render_gauge(value):
    color = "green" if value < 40 else "orange" if value < 70 else "red"
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = value,
        title = {'text': "Diabetes Risk %"},
        gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': color},
                 'steps': [{'range': [0, 40], 'color': "#e8f5e9"}, 
                           {'range': [40, 70], 'color': "#fff3e0"},
                           {'range': [70, 100], 'color': "#ffebee"}]}
    ))
    fig.update_layout(height=250, margin=dict(l=10,r=10,t=40,b=10))
    return fig

# ------------------ 4. PAGE LOGIC ------------------

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=60)
    st.title("DiaCare-Predict")
    st.write("Professional Diabetes Risk Assessment")
    st.markdown("---")
    
    # Progress Stepper
    steps = ["Gender", "Location", "Assessment"]
    curr = 0
    if st.session_state.page == "location": curr = 1
    if st.session_state.page == "predict": curr = 2
    
    for i, s in enumerate(steps):
        icon = "🟢" if i == curr else "✅" if i < curr else "⚪"
        st.write(f"{icon} **Step {i+1}: {s}**")
    
    st.markdown("---")
    # Change type to "primary"
    if st.button("🔄 Reset App", type="primary", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# --- PAGE 1: GENDER ---
if st.session_state.page == "gender":
    st.subheader("Step 1: Personal Details")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("Please select your biological sex for accurate medical referencing.")
        gender = st.radio("Select Gender", ("Female", "Male"), key="gender_select")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Next Step ➡️", type="primary"):
            st.session_state.gender = gender
            st.session_state.page = "location"
            st.rerun()
    with col2:
        st.info("ℹ️ **Why do we need this?**\nCertain health thresholds (like pregnancy history) differ biologically.")

# --- PAGE 2: LOCATION ---
elif st.session_state.page == "location":
    st.subheader("Step 2: Location Services")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("We use this to find specialized doctors near you.")
        method = st.radio("Preference:", ("Enter Location", "Skip Location"), horizontal=True)
        
        if method == "Enter Location":
            loc_input = st.text_input("City/Address", placeholder="e.g., New York, NY")
            
            if st.button("🔍 Verify Location"):
                with st.spinner("Locating..."):
                    coords = get_coordinates_from_location(loc_input)
                    if coords:
                        st.markdown(f'<div class="success-box">✅ Verified: {coords["display_name"]}</div>', unsafe_allow_html=True)
                        st.session_state.location = loc_input
                        st.session_state.coordinates = coords
                        st.session_state.location_verified = True
                    else:
                        st.error("❌ Location not found. Try a broader city name.")
                        st.session_state.location_verified = False
        
        st.markdown("---")
        c_back, c_next = st.columns([1, 1])
        if c_back.button("⬅️ Back"):
            st.session_state.page = "gender"
            st.rerun()
            
        # Continue Logic
        allow_next = (method == "Skip Location") or st.session_state.get('location_verified', False)
        if c_next.button("Continue to Assessment ➡️", type="primary", disabled=not allow_next):
            if method == "Skip Location":
                st.session_state.location = None
                st.session_state.coordinates = None
            st.session_state.page = "predict"
            st.rerun()

# --- PAGE 3: PREDICTION & RESULTS ---
elif st.session_state.page == "predict":
    st.subheader("Step 3: Medical Assessment")
    
    # Summary Bar
    st.info(f"👤 **Profile:** {st.session_state.gender} | 📍 **Location:** {st.session_state.get('location', 'Skipped')}")
    
    # --- INPUT FORM (3 Column Layout) ---
    with st.form("med_form"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("##### 🩸 Blood Metrics")
            glucose = st.number_input("Glucose (mg/dL)", 50, 300, 120, help="Fasting: 70-100")
            bp = st.number_input("Blood Pressure", 40, 200, 80, help="Diastolic")
            insulin = st.number_input("Insulin (µU/mL)", 0, 900, 80)

        with c2:
            st.markdown("##### 📏 Body Metrics")
            bmi = st.number_input("BMI", 10.0, 70.0, 25.0, step=0.1)
            skin = st.number_input("Skin Thickness", 0, 100, 20)
            age = st.number_input("Age (Years)", 1, 120, 30)

        with c3:
            st.markdown("##### 🧬 History")
            if st.session_state.gender == "Female":
                preg = st.number_input("Pregnancies", 0, 20, 1)
            else:
                preg = 0
                st.write("Pregnancies: N/A")
            
            dpf = st.number_input("Diabetes Pedigree", 0.0, 3.0, 0.5, step=0.01)
            
            st.markdown("---")
            fam_hist = st.checkbox("Family History")
            hyp_tens = st.checkbox("Hypertension")
            
        submitted = st.form_submit_button("🔍 ANALYZE RISK", type="primary", use_container_width=True)

    # --- RESULTS SECTION ---
    if submitted:
        # Prediction
        input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        prob = model.predict_proba(input_data)[0][1] * 100
        
        st.markdown("---")
        st.title("📊 Assessment Report")
        
        # 1. Gauge & Alert
        r_col1, r_col2 = st.columns([1, 2])
        with r_col1:
            st.plotly_chart(render_gauge(prob), use_container_width=True)
            
        with r_col2:
            if prob > 50:
                st.error(f"### ⚠️ HIGH RISK DETECTED")
                st.write("Your indicators suggest a strong likelihood of diabetes.")
                # Risk Analysis
                factors = []
                if glucose > 140: factors.append("High Glucose")
                if bmi > 30: factors.append("Obesity (High BMI)")
                if age > 45: factors.append("Age Factor")
                if fam_hist: factors.append("Family History")
                if factors: st.write("**Key Risk Drivers:** " + ", ".join(factors))
            else:
                st.success(f"### ✅ LOW RISK")
                st.write("Your indicators are within a healthy range. Keep maintaining your lifestyle!")

        st.markdown("---")
        
        # 2. Tabs for Details
        tab_doc, tab_diet, tab_tips, tab_data = st.tabs(["🏥 Find Doctors", "🍎 Diet Plan", "💡 Tips & Meds", "📋 Your Data"])
        
        with tab_doc:
            if st.session_state.get('coordinates'):
                st.write(f"Finding specialists near **{st.session_state.location}**...")
                coords = st.session_state.coordinates
                facilities = find_nearby_doctors(coords['lat'], coords['lon'], city=st.session_state.location.split(',')[0])
                
                if facilities:
                    for f in facilities:
                        # Professional HTML Card
                        verified_badge = "✅ Verified" if f.get('source') == 'database' else "🌐 Online Result"
                        map_url = f"http://maps.google.com/?q={f['lat']},{f['lon']}"
                        
                        st.markdown(f"""
                        <div class="doctor-card">
                            <div style="display:flex; justify-content:space-between;">
                                <div>
                                    <h4 style="margin:0;">{f['name']}</h4>
                                    <small style="color:gray;">{f['type']} • {f['specialty']} • ⭐ {f.get('rating', 'N/A')}</small><br>
                                    <span>📍 {f['address']}</span><br>
                                    <span>📞 {f['phone']}</span>
                                </div>
                                <div style="text-align:right; min-width:100px;">
                                    <span style="font-size:12px; background:#e0f7fa; padding:2px 6px; border-radius:4px;">{verified_badge}</span><br><br>
                                    <strong>{f['distance']} km</strong><br>
                                    <a href="{map_url}" target="_blank" style="text-decoration:none; color:#007bff;">Get Directions ↗</a>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("No specific facilities found via API. Try searching Google Maps.")
            else:
                st.info("⚠️ Location was skipped. General Hospital recommendations not available.")
                st.markdown("[🔍 Search Google for Endocrinologists Near Me](https://www.google.com/search?q=endocrinologist+near+me)")

        with tab_diet:
            d_tabs = st.tabs(diet_plan.keys())
            for i, (meal, items) in enumerate(diet_plan.items()):
                with d_tabs[i]:
                    st.write(f"**Recommended for {meal}:**")
                    for item in items:
                        st.write(f"• {item}")

        with tab_tips:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🏃 Lifestyle")
                for t in general_tips: st.write(f"• {t}")
            with c2:
                st.markdown("#### 💊 Common Meds")
                st.caption("Consult a doctor first.")
                for m in medications: st.write(f"• {m}")

        with tab_data:
            st.json({"Glucose": glucose, "BMI": bmi, "BP": bp, "Insulin": insulin, "Age": age})

# Footer
st.markdown("---")
st.caption("DiaCare AI | Educational Tool Only | Not a Substitute for Professional Medical Advice")