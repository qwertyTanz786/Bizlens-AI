# 📈 BizLens AI

<p align="center">
  <b>An AI-Powered Business Feasibility & Location Intelligence Platform for Dubai</b>
  <br><br>
  Analyze business ideas, predict success probabilities, and receive intelligent investment recommendations using Machine Learning and Large Language Models.
</p>

---

## 🚀 Overview

BizLens AI is an intelligent business decision support system designed to help entrepreneurs, investors, and startups evaluate the feasibility of opening a business in different communities across Dubai.

The platform combines **Machine Learning**, **Natural Language Processing**, and **Conversational AI** to transform a simple business idea into a detailed feasibility analysis.

Users simply describe their idea in plain English, and BizLens AI performs:

- Business Idea Understanding
- Community Detection
- Budget Extraction
- Competitor Analysis
- Success Probability Prediction
- AI-generated Business Consultation

---

## ✨ Features

### 🤖 AI Business Assistant

- Understands natural language
- Extracts business information automatically
- Detects Dubai communities
- Extracts investment budget
- Understands user intent
- Supports follow-up conversations

---

### 📊 Machine Learning Prediction

Predicts the probability of business success using engineered commercial features.

Outputs include:

- Success Probability
- Competition Level
- Infrastructure Analysis
- Business Suitability

---

### 📍 Dubai Community Intelligence

BizLens analyzes each Dubai community using multiple commercial indicators including:

- Restaurants
- Cafes
- Hotels
- Clinics
- Gyms
- Supermarkets
- Metro Stations
- Bus Stops
- Parking
- Walkability
- Office Presence
- Tourist Presence

---

### 💬 AI Business Consultant

After the ML prediction, the AI consultant generates:

- Executive Summary
- Business Opportunity
- Investment Advice
- Risk Analysis
- Practical Recommendations

---

## 🧠 System Workflow

```text
             User Input
                  │
                  ▼
       Large Language Model
        (Llama 3 + Ollama)
                  │
                  ▼
     Business Information Extraction
                  │
                  ▼
      Dubai Community Data Lookup
                  │
                  ▼
        Feature Engineering
                  │
                  ▼
     Random Forest ML Prediction
                  │
                  ▼
     Success Probability Score
                  │
                  ▼
      AI Business Recommendation
                  │
                  ▼
      Interactive Streamlit Dashboard
```

---

# 📊 Machine Learning

### Model

- Random Forest Classifier

### Feature Engineering

The prediction model considers factors such as:

- Business Type
- Commercial Rent Tier
- Unit Size
- Nearby Competitors
- Complementary Businesses
- Office Presence
- Residential Presence
- Tourist Presence
- Parking Availability
- Walkability
- Public Transport Access
- Development Maturity
- Competition Intensity
- Infrastructure Score

---

# 🤖 AI Components

BizLens uses **Llama 3** running locally through **Ollama**.

The language model is responsible for:

- Business Type Detection
- Community Detection
- Budget Extraction
- User Intent Classification
- Business Advice Generation
- Context-Aware Conversations

---

# 📂 Project Structure

```text
BizLens-AI
│
├── app.py
├── streamlit_app.py
├── config.py
│
├── core
│   ├── ml.py
│   └── llm.py
│
├── data
│   ├── historical_commercial_data.csv
│   └── dubai_community_metrics.csv
│
├── models
│   ├── bizlens_classifier.pkl
│   └── pipeline.pkl
│
├── requirements.txt
└── README.md
```

---

# 📊 Datasets

### Dubai Community Dataset

Contains verified commercial information for Dubai communities, including:

- Commercial Rent
- Restaurants
- Cafes
- Hotels
- Gyms
- Pharmacies
- Clinics
- Bus Stops
- Metro Stations
- Tourist Presence
- Office Presence
- Walkability
- Parking
- Development Maturity

---

### Historical Commercial Dataset

Training dataset used for the machine learning model containing:

- Business Type
- Community
- Commercial Rent Tier
- Unit Size
- Nearby Competitors
- Complementary Businesses
- Infrastructure Indicators
- Recommendation Labels

---

# ⚙ Installation

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

Download Ollama

https://ollama.com/download

Pull the Llama 3 model

```bash
ollama pull llama3
```

Start Ollama

```bash
ollama serve
```

---

# 🏋 Train the Machine Learning Model

```bash
python core/ml.py
```

This generates:

```text
models/
├── bizlens_classifier.pkl
└── pipeline.pkl
```

---

# ▶ Run BizLens AI

```bash
streamlit run streamlit_app.py
```

---

# 💬 Example

### User Input

```text
I have AED 450,000 and want to open a specialty coffee shop in Dubai Marina.
```

### BizLens AI Automatically

- Detects Coffee Shop
- Detects Dubai Marina
- Extracts Budget
- Calculates Nearby Competition
- Computes Infrastructure Score
- Predicts Business Success Probability
- Generates Executive Business Advice

---

# 📈 Sample Output(On Streamlit)

```
Business Type
Coffee Shop

Location
Dubai Marina

Budget
AED 450,000

Nearby Competitors
12

Success Probability
82.4%

Recommendation
Ollama generated Summary
```

---

# 🧪 Example Test Queries

```text
I want to open a gym in Business Bay.
```

```text
I have AED 800,000 and want to start a pharmacy in JVC.
```

```text
Would Downtown Dubai be suitable for a luxury restaurant?
```

```text
Suggest a better location for a bakery.
```

```text
Is the competition too high in Dubai Marina?
```

---

# 🛠 Technologies Used

| Category | Technology |
|-----------|------------|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas |
| Numerical Computing | NumPy |
| Visualization | Plotly |
| Frontend | Streamlit |
| Model Serialization | Joblib |
| LLM Runtime | Ollama |
| Language Model | Llama 3 |

---

# 🚀 Future Enhancements

- Interactive Dubai Map
- Rental Price Forecasting
- ROI Prediction
- Population Density Analysis
- Live Commercial Property Data
- Market Trend Analysis
- Business Ranking Engine
- Multi-City Support
- REST API
- Cloud Deployment

---

# 👨‍💻 Author

**Tanishq H. Panchal**

Computer Science Engineering Student

Machine Learning • Artificial Intelligence • Data Science

---

<p align="center">
Built with ❤️ using Python, Scikit-learn, Streamlit, and Llama 3.
</p>
