# 📈 BizLens AI

<div align="center">

### AI-Powered Business Feasibility & Location Intelligence for Dubai

**Machine Learning • Large Language Models • Business Analytics • Streamlit**

</div>

---

## 📖 Overview

BizLens AI is an intelligent business feasibility platform that helps entrepreneurs and investors evaluate whether a business idea is likely to succeed in a specific Dubai community.

The system combines **Machine Learning** and **Large Language Models (LLMs)** to understand natural language business ideas, analyze community characteristics, estimate competition, and generate professional investment recommendations.

Unlike traditional rule-based systems, BizLens AI leverages predictive analytics alongside conversational AI to provide explainable, data-driven business insights.

---

## ✨ Key Features

- 🤖 Natural Language Business Idea Analysis
- 🧠 Machine Learning Feasibility Prediction
- 📍 Dubai Community Intelligence
- 💬 AI Business Consultant powered by Ollama (Llama 3)
- 📊 Interactive Streamlit Dashboard
- 📈 Success Probability Prediction
- 🏢 Competitor Analysis
- 🚇 Infrastructure & Accessibility Evaluation
- 💰 Budget Extraction from User Input
- 📄 Executive Business Recommendations
- 💬 Context-Aware Follow-up Conversations

---

## 🎯 Problem Statement

Starting a business requires answering several critical questions:

- Is this location suitable?
- Is the competition too high?
- Is my budget sufficient?
- What are the risks?
- Is there enough customer demand?

Traditional feasibility studies are expensive and time-consuming.

BizLens AI provides entrepreneurs with an intelligent assistant capable of answering these questions within seconds using AI and Machine Learning.

---

# 🏗 System Architecture

```text
                User Input
                     │
                     ▼
        Natural Language Processing
            (Llama 3 via Ollama)
                     │
                     ▼
        Business Information Extraction
                     │
                     ▼
        Dubai Community Dataset Lookup
                     │
                     ▼
           Feature Engineering
                     │
                     ▼
      Random Forest Classification
                     │
                     ▼
       Business Success Probability
                     │
                     ▼
     AI Business Advisor (LLM Response)
                     │
                     ▼
        Interactive Streamlit Dashboard
```

---

# 🧠 AI Pipeline

### Step 1 — User Input

Example:

```text
I have AED 300,000 and want to open a specialty coffee shop in Dubai Marina.
```

---

### Step 2 — LLM Parser

The LLM extracts:

- Business Type
- Dubai Community
- Budget
- Risk Appetite
- Target Demographic
- User Intent

---

### Step 3 — Community Lookup

The extracted location is matched with the verified Dubai community dataset containing infrastructure and business statistics.

---

### Step 4 — Feature Engineering

Additional features are generated including:

- Competition Intensity
- Infrastructure Score
- Nearby Competitor Density

---

### Step 5 — Machine Learning

A Random Forest classifier predicts the probability of business success using engineered features.

---

### Step 6 — AI Advisor

The prediction is passed back to the LLM which generates a professional business recommendation based on:

- ML Prediction
- Business Type
- Budget
- Competition
- Community Characteristics

---

## 🧠 Machine Learning

### Algorithm

- Random Forest Classifier

### Feature Engineering

The model uses features such as:

- Business Type
- Commercial Rent Tier
- Commercial Unit Size
- Nearby Competitors
- Complementary Businesses
- Office Presence
- Residential Presence
- Tourist Presence
- Parking Availability
- Walkability
- Public Transport Access
- Development Maturity
- Infrastructure Score
- Competition Intensity

---

## 🤖 Large Language Model

BizLens AI uses **Llama 3** running locally through **Ollama**.

The LLM performs:

- Business Idea Parsing
- Location Detection
- Budget Extraction
- Intent Recognition
- Business Recommendation Generation
- Context-Aware Follow-up Conversations

---

# 📊 Datasets

## 1. Dubai Community Dataset

Contains verified information including:

- Communities
- Restaurants
- Cafes
- Hotels
- Gyms
- Clinics
- Pharmacies
- Schools
- Parks
- Bus Stops
- Metro Stations
- Commercial Rent Information
- Infrastructure Indicators

---

## 2. Business Training Dataset

Synthetic dataset containing:

- Business Type
- Community
- Rent Tier
- Unit Size
- Nearby Competitors
- Complementary Businesses
- Office Presence
- Residential Presence
- Tourist Presence
- Parking
- Walkability
- Development Maturity
- Recommendation Labels

---

# 📂 Project Structure

```text
BizLens-AI/
│
├── app.py
├── streamlit_app.py
├── config.py
│
├── core/
│   ├── ml.py
│   ├── llm.py
│
├── data/
│   ├── final_verified_dubai_communities_cleaned.csv
│   └── bizlens_dataset2_cleaned.csv
│
├── models/
│   ├── bizlens_classifier.pkl
│   └── pipeline.pkl
│
├── README.md
└── requirements.txt
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/BizLens-AI.git
cd BizLens-AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🦙 Install Ollama

Download Ollama:

https://ollama.com/download

Pull Llama 3

```bash
ollama pull llama3
```

Start Ollama

```bash
ollama serve
```

---

# 🏋️ Train the Machine Learning Model

```bash
python core/ml.py
```

Generated artifacts:

```text
models/
├── bizlens_classifier.pkl
└── pipeline.pkl
```

---

# ▶️ Run the Application

```bash
streamlit run streamlit_app.py
```

---

# 💬 Example Query

```text
I have AED 500,000 and want to open a specialty coffee shop in Dubai Marina.
```

---

# 📈 Example Output

- ✅ Business Type Detection
- ✅ Budget Extraction
- ✅ Dubai Community Detection
- ✅ Competitor Analysis
- ✅ Success Probability Prediction
- ✅ Interactive Gauge Visualization
- ✅ AI Investment Recommendation
- ✅ Context-Aware Follow-up Chat

---

# 🧪 Sample Test Cases

| User Input | Expected Output |
|------------|-----------------|
| Open a cafe in Dubai Marina | Feasibility prediction with AI recommendation |
| Open a pharmacy in JVC | Community analysis and success probability |
| Open a gym in Business Bay | Competitor analysis and prediction |
| I have AED 300,000 to invest | Budget extraction and personalized recommendation |
| Suggest a better location | Alternative Dubai community recommendation |
| Is this area too competitive? | Context-aware follow-up response |

---

# 📊 Tech Stack

| Category | Technology |
|-----------|------------|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| UI | Streamlit |
| Visualization | Plotly |
| Model Serialization | Joblib |
| LLM Runtime | Ollama |
| Language Model | Llama 3 |

---

# 🚀 Future Improvements

- Interactive Dubai Map
- Rental Price Prediction
- ROI Forecasting
- Population Density Analysis
- Live Market Data Integration
- Business Ranking Engine
- Multi-City Support
- Cloud Deployment
- Multiple LLM Support
- REST API Integration

---

# 👨‍💻 Author

**Tanishq H. Panchal**

Computer Science Engineering Student

Aspiring Data Scientist • Machine Learning Engineer • AI Enthusiast

---

## ⭐ If you found this project interesting, consider giving it a star!
