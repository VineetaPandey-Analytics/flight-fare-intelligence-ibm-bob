# ✈ Flight Fare Intelligence Dashboard

A clean, production-ready **Streamlit** analytics dashboard for India's domestic flight market — combining interactive data visualisation, grouped analytics, and a **Random Forest AI fare predictor** trained on 300,000+ real flight records. 

---

## 📸 Dashboard Preview

| Tab | Description |
|-----|-------------|
| 📊 **KPI Overview** | 7 high-level metrics + fare distribution charts |
| 📈 **Price Analytics** | Departure time, booking window, duration, stops & fare trend |
| 🗺️ **Route & Airline** | Airline rankings, route heatmap, Economy vs Business |
| 💡 **Insights & Recommendations** | 5 data insights + 2 business strategies |
| 🤖 **Fare Predictor** | Live AI fare estimation from user inputs |

---

## 🛠️ Tech Stack & Architecture

| Layer | Technology |
|-------|-----------|
| **Frontend (UI & Visuals)** | Streamlit, Plotly Express / Graph Objects, Custom CSS |
| **Backend & Logic** | Python 3.9+, Pandas, NumPy |
| **Machine Learning** | Scikit-learn (Random Forest Regressor) |
| **Data Source** | [Kaggle — Flight Fare Prediction Dataset](https://www.kaggle.com/code/varunsaikanuri/flight-fare-prediction-10-ml-models/input) |

---

## 📁 Project Structure

```
├── app.py                        # Main Streamlit application
├── Flight Fare Prediction.csv    # Source dataset (300,153 records)
├── Flight Fare Analytics.docx    # Project presentation (9 slides)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🗄️ Dataset

| Property | Value |
|----------|-------|
| **Source** | [Kaggle — Flight Fare Prediction Dataset](https://www.kaggle.com/code/varunsaikanuri/flight-fare-prediction-10-ml-models/input) |
| **Records** | 300,153 (after cleaning) |
| **Airlines** | 6 — IndiGo, Air India, Vistara, SpiceJet, AirAsia, GO FIRST |
| **Cities** | 6 — Delhi, Mumbai, Bangalore, Chennai, Hyderabad, Kolkata |
| **Classes** | Economy, Business |
| **Price Range** | ₹1,105 – ₹1,23,071 |

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `airline` | categorical | Airline name |
| `flight` | string | Flight code |
| `source_city` | categorical | Departure city |
| `departure_time` | categorical | Time slot (Early Morning / Morning / Afternoon / Evening / Night / Late Night) |
| `stops` | categorical | Number of stops (zero / one / two_or_more) |
| `arrival_time` | categorical | Arrival time slot |
| `destination_city` | categorical | Arrival city |
| `class` | categorical | Cabin class (Economy / Business) |
| `duration` | float | Flight duration in hours |
| `days_left` | int | Days remaining until departure |
| `price` | int | Fare in Indian Rupees (₹) — **target variable** |

---

## ⚙️ Data Pipeline

The [`load_data()`](app.py) function runs a full preprocessing pipeline on every startup (cached via `@st.cache_data`):

1. **Load** — read CSV, strip/normalise column names
2. **Deduplicate** — drop exact duplicate rows
3. **Type coercion** — cast `price`, `duration`, `days_left` to numeric; drop rows with nulls on critical columns
4. **Filter** — remove fares below ₹500 (data entry errors)
5. **Feature engineering**
   - `departure_hour` — numeric hour mapped from time-slot labels
   - `booking_window` — 5-bucket categorical (0–7 / 8–14 / 15–30 / 31–60 / 60+ days)
   - `route` — `"source_city → destination_city"` string
   - `duration_bucket` — 4-bucket categorical (< 2 hrs / 2–4 hrs / 4–8 hrs / 8+ hrs)
   - `stops_num` — integer encoding of stops (0 / 1 / 2)

---

## 📊 Dashboard Tabs

### 📊 Tab 1 — KPI Overview
- **Row 1:** Total Flights · Average Fare · Max Fare · Most Popular Route
- **Row 2:** Economy Share · Non-stop % · Median Days Left
- Fare frequency histogram (60 bins)
- Average Fare by Cabin Class bar chart with ₹ data labels

### 📈 Tab 2 — Price Analytics
- **Departure Time** — bar + line combo showing avg & median fare per time slot
- **Booking Window** — colour-gradient bars for 5 advance-booking buckets
- **Fare vs Days Left** — 4,000-record scatter coloured by airline
- **Duration & Stops** — duration-bucket bars + Economy vs Business stops comparison
- **Rolling Trend** — full-width area line chart (x-axis reversed, 5-period rolling avg)

### 🗺️ Tab 3 — Route & Airline
- Airline avg fare ranked bar chart
- Flight volume share donut chart
- 6×6 Source → Destination price heatmap
- Economy vs Business grouped bar chart per airline
- Top 10 routes by volume (sortable table with min / avg / max fare)

### 💡 Tab 4 — Insights & Recommendations
**5 Data Insights** (computed live from filtered data):
1. Advance Booking Saves Money (~113% last-minute premium)
2. Departure Time Drives Price (Early Morning cheapest)
3. Non-stop Convenience Premium
4. Airline Price Spread (2× between AirAsia and Vistara)
5. Business Class ~400% premium over Economy

**2 Business Recommendations:**
- 🔶 **Strategy 1** — Tiered advance-booking dynamic pricing ladder (10–15% discount at 45+ days, 1.2–2.0× surge within 14 days)
- 🟢 **Strategy 2** — Peak-hour surcharge (₹500–₹1,500) + off-peak incentive programme (₹300–₹700 discount)

### 🤖 Tab 5 — AI Fare Predictor
Enter 8 inputs → get an instant AI-estimated fare with historical comparison.

| Input | Control |
|-------|---------|
| Airline | Dropdown |
| Cabin Class | Dropdown |
| Source City | Dropdown |
| Destination City | Dropdown |
| Departure Time | Dropdown |
| Stops | Dropdown |
| Days Left | Slider (1–49) |
| Flight Duration | Slider (1.0–20.0 hrs) |

**Output:** Predicted fare (₹) + delta vs historical average for the same route/airline/class.

---

## 🤖 Machine Learning Model

| Property | Value |
|----------|-------|
| **Algorithm** | Random Forest Regressor (`sklearn`) |
| **Estimators** | 120 trees |
| **Max Depth** | 18 |
| **Min Samples Leaf** | 4 |
| **Train / Test Split** | 85% / 15% |
| **MAE** | ~₹2,161 |
| **R² Score** | ~0.97 |

All categorical features are label-encoded before training. The model is cached via `@st.cache_resource` and trained once on app startup.

---

## 🎨 Design System

| Token | Hex | Usage |
|-------|-----|-------|
| Navy | `#1B2A4A` | Primary background, header, KPI cards |
| Teal | `#2E8B7A` | Accent bars, chart primary series, tags |
| Slate | `#4A6FA5` | Secondary charts, insight borders |
| Accent | `#F4A261` | Recommendation borders, callouts |
| Light | `#F0F4FA` | Page background, info panels |

Custom CSS overrides include:
- Dark navy sidebar with slate tag pills (replaces Streamlit's default red)
- Styled tab bar (active tab: navy bg / white text)
- KPI cards with teal left accent border and drop shadow
- Currency-formatted Plotly chart axes (`₹xx,xxx`)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# 1. Clone or download the project
git clone <your-repo-url>
cd flight-fare-dashboard

# 2. (Recommended) Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the dataset in the project root
#    File: Flight Fare Prediction.csv

# 5. Launch the dashboard
# Launch the dashboard
```
python -m streamlit run app.py
```
# Alternatively, you can run:
python -m streamlit run VineetaPandey_FlightFareIntelligence.py
```

The app will open at **http://localhost:8501** in your default browser.

### Dependencies

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.26.0
plotly>=5.20.0
scikit-learn>=1.4.0
```

> **Note:** The `trendline="lowess"` option is **not used** in this project. `statsmodels` is therefore not required.

---

## 🔍 Sidebar Filters

All five tabs react live to the following filters:

| Filter | Type | Options |
|--------|------|---------|
| **Airlines** | Multi-select | IndiGo, Air India, Vistara, SpiceJet, AirAsia, GO FIRST |
| **Cabin Class** | Multi-select | Economy, Business |
| **Days Left to Departure** | Range slider | 1 – 49 |
| **Source City** | Multi-select | Delhi, Mumbai, Bangalore, Chennai, Hyderabad, Kolkata |

---

## ✨ Key Features

- **Interactive Analytics Dashboard** — 5 dedicated tabs covering KPI overview, price analytics, route & airline breakdowns, insights, and AI fare prediction
- **AI Fare Predictor** — Random Forest model (R² ~0.97) trained on 300,000+ records for instant fare estimation from 8 user inputs
- **Live Market Verification** — Integrated dynamic, parameter-based redirection buttons to Google Flights for real-time market price cross-checking
- **Global Sidebar Filters** — Airline, cabin class, days-left range, and source city filters applied live across all tabs
- **Custom Design System** — Navy/teal/slate palette with styled KPI cards, tab bar, and currency-formatted Plotly axes

---

## 📌 Key Findings

- **Last-minute bookings** cost ~113% more than 30+ day advance purchases
- **Early Morning** departures are consistently the cheapest time slot across all routes
- **Vistara** is the most expensive airline; **AirAsia** the most affordable (2× spread)
- **Business class** averages ~4–5× Economy fare on domestic Indian routes
- Fares follow a **U-curve** vs days-to-departure: lowest 20–35 days out, spiking near day 1

---

## 📄 License

This project is for educational and portfolio purposes. Dataset credits to its original authors.

---

<div align="center">
  Built with Python · Streamlit · Plotly · scikit-learn
</div>
