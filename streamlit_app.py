import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from core.ml import BizLensPredictor, load_community_dataset
from core.llm import BusinessParser, BusinessAdvisor
from config import MODEL_PATH, PIPELINE_PATH
import time

st.set_page_config(page_title="BizLens AI", page_icon="📈", layout="wide")

# Custom CSS for a premium feel
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Outfit:wght@500;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    /* Hide Streamlit default header and footer */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stApp {
        background-color: #000000;
        background-image: radial-gradient(circle at 15% 50%, rgba(59, 130, 246, 0.15), transparent 25%),
                          radial-gradient(circle at 85% 30%, rgba(139, 92, 246, 0.15), transparent 25%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
    }
    
    .hero-container {
        text-align: center;
        padding: 3rem 0 2rem 0;
        animation: fadeIn 1s ease-in-out;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: #94a3b8;
        font-weight: 300;
    }

    .metric-container {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 25px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(59, 130, 246, 0.2);
        border-color: rgba(59, 130, 246, 0.3);
    }
    
    .metric-title {
        color: #94a3b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(to right, #e2e8f0, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .advisor-box {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 30px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-left: 5px solid #8b5cf6;
        margin-top: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        transition: transform 0.3s ease;
        line-height: 1.6;
        color: rgba(248, 250, 252, 0.9);
    }
    
    .advisor-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 40px rgba(139, 92, 246, 0.2);
    }
    
    .advisor-box h3 {
        color: #e2e8f0;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Animation keyframes */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Input styling override */
    div[data-baseweb="input"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
    }
    
    div[data-baseweb="input"]:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# Cache data and models
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    community_df = load_community_dataset()
    predictor = BizLensPredictor(model, pipeline)
    parser = BusinessParser()
    advisor = BusinessAdvisor()
    valid_locations = community_df['community_name'].dropna().astype(str).unique().tolist()
    return predictor, parser, advisor, community_df, valid_locations

try:
    predictor, parser, advisor, community_df, valid_locations = load_assets()
except Exception as e:
    st.error(f"Failed to load models or data: {e}")
    st.stop()

def get_community_data(community_name):
    if not community_name:
        return None
    match = community_df[community_df['community_name'].astype(str).str.contains(community_name, case=False, na=False)]
    if not match.empty:
        return match.iloc[0].fillna(0) 
    return None

def create_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability,
        number = {"suffix": "%", "valueformat": ".1f"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Success Probability", 'font': {'size': 24, 'color': '#94a3b8'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#334155"},
            'bar': {'color': "#3b82f6"},
            'bgcolor': "rgba(30, 41, 59, 0.7)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.1)",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(239, 68, 68, 0.3)'},
                {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [70, 100], 'color': 'rgba(16, 185, 129, 0.3)'}],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': probability}
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def render_analysis_ui(data, key=None):
    has_budget = bool(data.get('Budget'))
    cols = st.columns(4) if has_budget else st.columns(3)
    
    with cols[0]:
        st.markdown(f"<div class='metric-container'><div class='metric-title'>📍 Location</div><div class='metric-value'>{data['Target_Location']}</div></div>", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"<div class='metric-container'><div class='metric-title'>🏪 Business</div><div class='metric-value'>{data['Business_Type']}</div></div>", unsafe_allow_html=True)
    with cols[2]:
        if has_budget:
            st.markdown(f"<div class='metric-container'><div class='metric-title'>💰 Budget</div><div class='metric-value'>{data['Budget']}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='metric-container'><div class='metric-title'>👥 Competitors</div><div class='metric-value'>{data['Competitors']}</div></div>", unsafe_allow_html=True)
    if has_budget:
        with cols[3]:
            st.markdown(f"<div class='metric-container'><div class='metric-title'>👥 Competitors</div><div class='metric-value'>{data['Competitors']}</div></div>", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    g_col1, g_col2 = st.columns([1, 1.5])
    
    with g_col1:
        fig = create_gauge(data['Probability'])
        st.plotly_chart(fig, use_container_width=True, key=key)
        
    with g_col2:
        st.markdown(f"""
        <div class='advisor-box'>
            <h3>🤖 AI Real Estate Advisor</h3>
            <p>{data['Advice'].replace(chr(10), '<br>')}</p>
        </div>
        """, unsafe_allow_html=True)


# UI Layout
st.markdown("""
<div class="hero-container">
    <div class="hero-title"><i class="fa-solid fa-chart-line"></i> BizLens AI</div>
    <div class="hero-subtitle">Your Intelligent Real Estate Business Feasibility Consultant</div>
</div>
""", unsafe_allow_html=True)

# State Management Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None

# Render Chat History
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg.get("type") == "analysis":
            render_analysis_ui(msg["data"], key=f"history_analysis_{i}")
        elif msg.get("type") == "advisor_html":
            st.markdown(msg["content"], unsafe_allow_html=True)
        else:
            st.write(msg["content"])

# Chat Input Handler
user_idea = st.chat_input("Share your business vision...")

if user_idea:
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": user_idea})
    with st.chat_message("user"):
        st.write(user_idea)
    
    with st.spinner("Analyzing..."):
        parsed = parser.parse(user_idea, valid_locations=valid_locations)
        
        target_loc = parsed.get("Target_Location", "").strip()
        is_valid = parsed.get("Is_Valid_Idea", True)
        
        # Fallback for location extraction
        if is_valid and not target_loc:
            for loc in community_df['community_name']:
                if str(loc).lower() in user_idea.lower():
                    target_loc = loc
                    break
        
        # --- PATH A: New Business Idea ---
        if is_valid and target_loc:
            row = get_community_data(target_loc)
            if row is not None:
                business_type_raw = parsed.get("Business_Type", "Retail").lower()
                
                # Competitor calculation
                community_competitors = 0
                if "cafe" in business_type_raw:
                    community_competitors = row.get("cafe_count", 0)
                elif "restaurant" in business_type_raw:
                    community_competitors = row.get("restaurant_count", 0)
                elif "gym" in business_type_raw or "fitness" in business_type_raw:
                    community_competitors = row.get("gym_count", 0)
                elif "pharmacy" in business_type_raw or "medical" in business_type_raw:
                    community_competitors = row.get("pharmacy_count", 0) + row.get("clinic_count", 0)
                elif "supermarket" in business_type_raw or "grocery" in business_type_raw:
                    community_competitors = row.get("supermarket_count", 0)
                else:
                    community_competitors = row.get("cafe_count", 0)

                nearby_competitors = min(int(community_competitors * 0.15), 35)
                
                # Complementary logic
                community_complementary = int(row.get("hotel_count", 0) + row.get("bus_stop_count", 0) + row.get("metro_station_count", 0) * 5)
                comp_category = "Residential"
                if "clinic" in business_type_raw or "pharmacy" in business_type_raw or "medical" in business_type_raw:
                    comp_category = "Healthcare"
                elif community_complementary > 20:
                    comp_category = "Mixed Retail"
                elif community_complementary > 8:
                    comp_category = "Office & Retail"

                business_type_formatted = business_type_raw.title()

                input_series = pd.Series({
                    "Community": target_loc,
                    "Business_Type": business_type_formatted,
                    "Commercial_Rent_Tier": row.get("Commercial_Rent_Tier", "Medium"),
                    "Commercial_Unit_Size_sqm": row.get("Commercial_Unit_Size_sqm", 100),
                    "Nearby_Competitors": nearby_competitors,
                    "Nearby_Complementary_Businesses": comp_category,
                    "Office_Presence": row.get("Office_Presence", "Medium"),
                    "Residential_Presence": row.get("Residential_Presence", "Medium"),
                    "Tourist_Presence": row.get("Tourist_Presence", "Medium"),
                    "Parking_Availability": row.get("Parking_Availability", "Medium"),
                    "Walkability": row.get("Walkability", "Medium"),
                    "Street_Visibility": row.get("Street_Visibility", "Medium"),
                    "Public_Transport_Access": row.get("Public_Transport_Access", "Medium"),
                    "Development_Maturity": row.get("Development_Maturity", "Mature")
                })

                result = predictor.predict(input_series)
                advice = advisor.audit(parsed, result, target_loc, user_idea)
                
                analysis_data = {
                    "Target_Location": target_loc,
                    "Business_Type": business_type_formatted,
                    "Competitors": nearby_competitors,
                    "Probability": result['probability'],
                    "Advice": advice,
                    "Budget": parsed.get("Budget")
                }
                
                # Update State
                st.session_state.current_analysis = analysis_data
                st.session_state.messages.append({"role": "assistant", "type": "analysis", "data": analysis_data, "content": "Analysis Complete."})
                
                with st.chat_message("assistant"):
                    render_analysis_ui(analysis_data, key=f"new_analysis_{len(st.session_state.messages)}")
            else:
                # Location found but not in DB
                advice = advisor.handle_general_query(user_idea)
                html_content = f"<div class='advisor-box'><h3>🤖 AI Real Estate Advisor</h3><p>{advice.replace(chr(10), '<br>')}</p></div>"
                st.session_state.messages.append({"role": "assistant", "type": "advisor_html", "content": html_content})
                with st.chat_message("assistant"):
                    st.markdown(html_content, unsafe_allow_html=True)

        # --- PATH B: Follow-up Chat ---
        elif is_valid and not target_loc and st.session_state.current_analysis:
            # We have an ongoing session, no new location was detected, so it's a follow-up question
            advice = advisor.handle_followup(user_idea, st.session_state.messages, st.session_state.current_analysis)
            html_content = f"<div class='advisor-box'><h3>💬 AI Follow-up</h3><p>{advice.replace(chr(10), '<br>')}</p></div>"
            st.session_state.messages.append({"role": "assistant", "type": "advisor_html", "content": html_content})
            with st.chat_message("assistant"):
                st.markdown(html_content, unsafe_allow_html=True)
                
        # --- PATH C: Unknown/Negative/No Context ---
        else:
            advice = advisor.handle_general_query(user_idea)
            html_content = f"<div class='advisor-box'><h3>🤖 AI Real Estate Advisor</h3><p>{advice.replace(chr(10), '<br>')}</p></div>"
            st.session_state.messages.append({"role": "assistant", "type": "advisor_html", "content": html_content})
            with st.chat_message("assistant"):
                st.markdown(html_content, unsafe_allow_html=True)
