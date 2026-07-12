import joblib
import pandas as pd
from core.ml import BizLensPredictor, load_community_dataset
from core.llm import BusinessParser, BusinessAdvisor
from config import MODEL_PATH, PIPELINE_PATH

def get_community_data(community_name, community_df):
    if not community_name:
        return None
    match = community_df[community_df['community_name'].astype(str).str.contains(community_name, case=False, na=False)]
    
    if not match.empty:
        return match.iloc[0].fillna(0) 
    return None

def main():
    # 1. Load artifacts and data
    try:
        model = joblib.load(MODEL_PATH)
        pipeline = joblib.load(PIPELINE_PATH)
    except FileNotFoundError:
        print("Model artifacts not found. Please run trainer.py first.")
        return

    community_df = load_community_dataset()
    
    predictor = BizLensPredictor(model, pipeline)
    parser = BusinessParser()
    advisor = BusinessAdvisor()

    # 2. Get User Input
    print("Welcome to BizLens AI!")
    user_idea = input("Describe your business idea in a normal sentence:\n> ")
    valid_locations = community_df['community_name'].dropna().astype(str).unique().tolist()
    parsed = parser.parse(user_idea, valid_locations=valid_locations)
    
    # 3. Dynamic Lookup
    target_loc = parsed.get("Target_Location", "").strip()
    is_valid = parsed.get("Is_Valid_Idea", True)
    
    # Fallback to simple string matching ONLY if the idea is valid but the LLM parser missed the location
    if is_valid and not target_loc:
        for loc in community_df['community_name']:
            if str(loc).lower() in user_idea.lower():
                target_loc = loc
                break
                
    if not is_valid or not target_loc:
        row = None
    else:
        row = get_community_data(target_loc, community_df)
    
    if row is None:
        reason = "Invalid idea or negative intent" if not is_valid else f"Unknown or invalid location '{target_loc if target_loc else 'Not provided'}'"
        print(f"\n[ML Prediction] Skipped: {reason}.")
        print("Generating expert business advice...")
        advice = advisor.handle_general_query(user_idea)
        print("\n--- Real Estate Business Advisor ---")
        print(advice)
        print("------------------------------------\n")
        return

    # Compute competitors based on business type and available community_df counts
    business_type_raw = parsed.get("Business_Type", "Retail").lower()
    
    # Match specific competitors to the business type
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
        community_competitors = row.get("cafe_count", 0) # generic fallback

    # Scale macro community counts down to micro "Nearby" counts to match ML training distribution (mean=7, max=50)
    nearby_competitors = min(int(community_competitors * 0.15), 35)
    
    community_complementary = int(row.get("hotel_count", 0) + row.get("bus_stop_count", 0) + row.get("metro_station_count", 0) * 5)
    
    # Map to categorical Nearby_Complementary_Businesses expected by the ML model
    comp_category = "Residential"
    if "clinic" in business_type_raw or "pharmacy" in business_type_raw or "medical" in business_type_raw:
        comp_category = "Healthcare"
    elif community_complementary > 20:
        comp_category = "Mixed Retail"
    elif community_complementary > 8:
        comp_category = "Office & Retail"

    business_type_formatted = business_type_raw.title()

    # 4. Map the data from the CSV to the model's expected input
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

    # 5. Predict
    result = predictor.predict(input_series)
    print(f"\n[ML Prediction] Feasibility Probability for {target_loc}: {result['probability']}%\n")

    # 6. Advisor Prompt via Ollama
    print("Generating expert business advice based on the prediction...")
    advice = advisor.audit(parsed, result, target_loc, user_idea)
    print("\n--- Real Estate Business Advisor ---")
    print(advice)
    print("------------------------------------\n")

if __name__ == "__main__":
    main()