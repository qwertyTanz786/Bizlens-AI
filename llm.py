import json
import requests
import re
from dataclasses import dataclass
from typing import List
from config import AI_API_URL, AI_MODEL_NAME

# --- Parser ---

class BusinessParser:
    def __init__(self, model="llama3"):
        self.model = model

    def parse(self, user_input: str, valid_locations: list = None) -> dict:
        locations_prompt = ""
        if valid_locations:
            locations_list = ", ".join(valid_locations)
            locations_prompt = f"""
            CRITICAL LOCATION RULES: 
            1. If the user mentions a location, you MUST map it to one of these exact names: [{locations_list}].
            2. If the user mentions a landmark, map it to the broader community.
            3. IF THE USER DOES NOT MENTION A SPECIFIC LOCATION, LEAVE "Target_Location" COMPLETELY EMPTY (""). DO NOT GUESS.
            """

        prompt = f"""
        You are an expert Business Consultant. Your task is to analyze the user's idea and extract the profile into JSON.
        Idea: '{user_input}'
        {locations_prompt}
        
        CRITICAL INTENT RULES: 
        1. If the user explicitly states they do NOT want to open a business, or their input is completely unrelated to business, set "Is_Valid_Idea" to false.
        2. If the user asks a general business question, proposes an idea, or mentions a budget/price, set "Is_Valid_Idea" to true.
        3. Only extract the EXACT business type mentioned by the user (e.g. if they say "bakery", do NOT output "cafe").

        JSON Schema:
        {{
            "User_Intent_Reasoning": "Briefly explain what the user is asking.",
            "Is_Valid_Idea": true or false,
            "Target_Location": "Extracted community name, or empty string if not mentioned",
            "Business_Type": "Extracted business type, or empty string if not mentioned",
            "Budget": "Extracted numerical budget or price if mentioned (e.g., '100,000 AED', '$50k'), or null if not mentioned",
            "Budget_Tier": "Low/Medium/High (infer from budget if provided)",
            "Risk_Appetite": "Low/Medium/High",
            "Target_Demographic": "e.g. Corporate/Tourists/Residents"
        }}
        Return ONLY valid JSON. Do not include markdown code blocks.
        """
        try:
            response = requests.post("http://localhost:11434/api/generate", json={
                "model": self.model, "prompt": prompt, "format": "json", "stream": False
            }, timeout=60)
            text_response = response.json().get('response', '{}')
            # Try to strip markdown formatting if the model included it
            text_response = re.sub(r'```json\s*', '', text_response)
            text_response = re.sub(r'```\s*', '', text_response)
            return json.loads(text_response)
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return {"Target_Location": "", "Business_Type": "Retail"}


# --- Advisor ---

@dataclass
class AdvisoryReport:
    executive_summary: str
    opportunity: str
    competition: str
    risk_level: str
    strengths: List[str]
    recommendations: List[str]
    conclusion: str

class BusinessAdvisor:
    def audit(self, profile, ml_result, community_name, user_idea=""):
        prob = ml_result.get('probability', 0)
        
        budget = profile.get('Budget')
        budget_text = f"- Budget: {budget}" if budget else ""
        
        prompt = f"""
        You are a highly experienced and savvy real estate business consultant specializing in commercial properties in {community_name}.
        A client just approached you with this statement: "{user_idea}"
        
        Based on this, you extracted the following business profile:
        - Business Type: {profile.get('Business_Type', 'Retail')}
        - Location: {community_name}
        {budget_text}
        
        Based on our Machine Learning models, the success probability for this venture is: {prob}%
        
        Provide a descriptive, engaging, and professional response just like a real estate business person talking to a client. 
        - If the probability is >= 70%: Express excitement about the prime location and scalability.
        - If the probability is 40-69%: Give practical advice on operational efficiency, securing good rent, and optimizing the business.
        - If the probability is < 40%: Be honest but professional about the high risks, suggesting a pivot in location or business type.
        
        Write in a conversational, descriptive manner. Deliver a 3-paragraph executive summary. Do not use generic letter salutations.
        """
        
        try:
            response = requests.post(AI_API_URL, json={
                "model": AI_MODEL_NAME,
                "prompt": prompt,
                "stream": False
            }, timeout=60)
            return response.json().get("response", "Analysis unavailable.")
        except Exception as e:
            return f"Strategic analysis temporarily unavailable: {str(e)}"
            
    def handle_general_query(self, user_idea):
        prompt = f"""
        You are a highly experienced and savvy real estate business consultant in the UAE.
        A client just said: "{user_idea}"
        
        If they asked a general business, budget, or real estate question, provide a helpful and professional answer.
        If they proposed a business but didn't specify a valid Dubai community, give them some general advice on their idea and ask them which area they are considering (e.g., Downtown Dubai, Marina, etc.).
        If they provided a location that doesn't exist in your records, inform them of that and suggest a few popular ones.
        If their input is completely unrelated to business or explicitly negative, politely steer them back to real estate and business consulting.
        
        Deliver a professional, engaging, and slightly witty response in 2-3 paragraphs. Do not use generic letter salutations.
        """
        
        try:
            response = requests.post(AI_API_URL, json={
                "model": AI_MODEL_NAME,
                "prompt": prompt,
                "stream": False
            }, timeout=60)
            return response.json().get("response", "Response unavailable.")
        except Exception as e:
            return f"Service temporarily unavailable: {str(e)}"
            
    def generate(self, recommendation) -> AdvisoryReport:
        # Dummy generation method to satisfy report.py dependencies
        return AdvisoryReport(
            executive_summary=f"Summary for {recommendation.community}",
            opportunity="High",
            competition="Moderate",
            risk_level="Low",
            strengths=["Location", "Traffic"],
            recommendations=["Open soon", "Marketing"],
            conclusion="Good prospect"
        )
        
    def handle_followup(self, user_idea, chat_history, context):
        history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-5:] if msg['role'] in ['user', 'assistant']])
        
        prompt = f"""
        You are a highly experienced and savvy real estate business consultant in the UAE.
        You are currently engaged in a conversation with a client about their business idea.
        
        Active Business Context:
        - Business Type: {context.get('Business_Type', 'N/A')}
        - Location: {context.get('Target_Location', 'N/A')}
        - ML Success Probability: {context.get('Probability', 'N/A')}%
        - Nearby Competitors: {context.get('Competitors', 'N/A')}
        
        Recent Chat History:
        {history_text}
        
        Client's New Message: "{user_idea}"
        
        Provide a helpful, direct, and expert response to their new message. Keep it conversational but highly professional. 
        If they ask about something unrelated to real estate or business, gently steer them back.
        Do not use generic letter salutations.
        """
        
        try:
            response = requests.post(AI_API_URL, json={
                "model": AI_MODEL_NAME,
                "prompt": prompt,
                "stream": False
            }, timeout=60)
            return response.json().get("response", "Response unavailable.")
        except Exception as e:
            return f"Service temporarily unavailable: {str(e)}"
