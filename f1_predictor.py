import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from data_loader import F1DataLoader
import time
from streamlit_lottie import st_lottie
import json
import requests

# Calendario F1 2025
F1_CALENDAR_2025 = [
    "Australian Grand Prix - Melbourne (16 Mar) - COMPLETED",
    "Chinese Grand Prix - Shanghai (23 Mar) - COMPLETED",
    "Japanese Grand Prix - Suzuka (6 Apr)",
    "Bahrain Grand Prix - Sakhir (13 Apr)",
    "Saudi Arabian Grand Prix - Jeddah (20 Apr)",
    "Miami Grand Prix - Miami (4 May)",
    "Emilia Romagna Grand Prix - Imola (18 May)",
    "Monaco Grand Prix - Monte Carlo (25 May)",
    "Spanish Grand Prix - Barcelona (1 Jun)",
    "Canadian Grand Prix - Montreal (15 Jun)",
    "Austrian Grand Prix - Spielberg (29 Jun)",
    "British Grand Prix - Silverstone (6 Jul)",
    "Belgian Grand Prix - Spa-Francorchamps (27 Jul)",
    "Hungarian Grand Prix - Budapest (3 Aug)",
    "Dutch Grand Prix - Zandvoort (31 Aug)",
    "Italian Grand Prix - Monza (7 Sep)",
    "Azerbaijan Grand Prix - Baku (21 Sep)",
    "Singapore Grand Prix - Singapore (5 Oct)",
    "United States Grand Prix - Austin (19 Oct)",
    "Mexico City Grand Prix - Mexico City (26 Oct)",
    "São Paulo Grand Prix - São Paulo (9 Nov)",
    "Las Vegas Grand Prix - Las Vegas (22 Nov)",
    "Qatar Grand Prix - Lusail (30 Nov)",
    "Abu Dhabi Grand Prix - Yas Marina (7 Dec)"
]

# Team colors for consistent styling
TEAM_COLORS = {
    'Red Bull Racing': '#0600EF',
    'Ferrari': '#DC0000',
    'Mercedes': '#00D2BE',
    'McLaren': '#FF8700',
    'Aston Martin': '#006F62',
    'Alpine': '#0090FF',
    'Williams': '#005AFF',
    'RB': '#2B4562',
    'Kick Sauber': '#52E252',
    'Haas F1 Team': '#FFFFFF'
}

# UI Enhancement - Sidebar
st.set_page_config(page_title="F1 ORACLE", page_icon="🏎️", layout="wide")
st.sidebar.title("⚙️ App Info")
st.sidebar.markdown("""
**F1 ORACLE**

Predict the 2025 Formula 1 season using machine learning.

- Model: Random Forest
- Data: Historical F1 Data
- Dev: Shashwat Kashyap

[GitHub Repo](https://github.com/Shashwat-k-9114/F1-ORACLE)
""")

# Placeholder - The rest of your simulation and driver input logic continues below as-is

class F1Predictor:
    def __init__(self, data_path='f1data'):
        self.data_path = data_path
        self.model = None
        self.data_loader = F1DataLoader(data_path)
        self.feature_importance = None
        self.grid_2025 = None
        self.results_2025 = None
        self.load_2025_data()
        # Try to load existing model at initialization
        try:
            self.load_model()
        except:
            pass
    
    def load_2025_data(self):
        """Load the 2025 F1 grid and results data."""
        try:
            self.grid_2025 = pd.read_csv(f'{self.data_path}/f1_2025_grid.csv')
            self.results_2025 = pd.read_csv(f'{self.data_path}/f1_2025_results.csv')
        except Exception as e:
            print(f"Error loading 2025 data: {e}")
    
    def get_driver_recent_results(self, driver_name):
        """Get recent results for a driver in 2025."""
        if self.results_2025 is None:
            return None
        
        driver_results = self.results_2025[self.results_2025['driver_name'] == driver_name]
        return driver_results.sort_values('date', ascending=False)
    
    def predict_2025_race(self, circuit_name, qualifying_results=None):
        """
        Predict the outcome of a 2025 race.
        
        Args:
            circuit_name (str): Name of the circuit
            qualifying_results (dict): Optional dictionary with grid positions for each driver
        """
        if self.model is None or self.grid_2025 is None:
            return None
            
        # Create a prediction dataframe
        pred_df = self.grid_2025.copy()
        
        # Add default values for required features
        pred_df['grid'] = range(1, len(pred_df) + 1)  # Default grid positions
        if qualifying_results:
            for driver_id, position in qualifying_results.items():
                pred_df.loc[pred_df['driverId'] == driver_id, 'grid'] = position
        
        # Add other required features with reasonable default values
        pred_df['qual_position_avg'] = pred_df['grid']
        
        # Update points_moving_avg based on 2025 results
        pred_df['points_moving_avg'] = 0
        if self.results_2025 is not None:
            for idx, row in pred_df.iterrows():
                recent_results = self.get_driver_recent_results(row['driver_name'])
                if recent_results is not None and not recent_results.empty:
                    # Calculate points based on positions (simplified)
                    points = recent_results['position'].map(lambda x: max(26-x, 0)).mean()
                    pred_df.loc[idx, 'points_moving_avg'] = points
        
        pred_df['circuit_wins'] = 0  # Could be updated with historical data
        pred_df['points_championship'] = pred_df['points_moving_avg']
        
        # Calculate championship positions based on points
        pred_df['position_championship'] = pred_df['points_championship'].rank(ascending=False, method='min')
        
        # Calculate constructor stats
        constructor_stats = pred_df.groupby('team_name').agg({
            'points_moving_avg': ['mean', 'std']
        }).reset_index()
        constructor_stats.columns = ['team_name', 'constructor_points_mean', 'constructor_points_std']
        
        pred_df = pd.merge(pred_df, constructor_stats, on='team_name', how='left')
        pred_df['constructor_position_mean'] = pred_df['constructor_points_mean'].rank(ascending=False, method='min')
        
        # Encode categorical variables
        for col, encoder in self.data_loader.label_encoders.items():
            if col == 'nationality':
                pred_df[f'{col}_encoded'] = encoder.transform(pred_df['nationality'])
            elif col == 'nationality_constructor':
                pred_df[f'{col}_encoded'] = encoder.transform(pred_df['constructor_nationality'])
            elif col == 'country':
                # Use a default value for now
                pred_df[f'{col}_encoded'] = 0
        
        # Select features in the same order as training
        feature_columns = [
            'grid',
            'qual_position_avg',
            'points_moving_avg',
            'circuit_wins',
            'points_championship',
            'position_championship',
            'constructor_points_mean',
            'constructor_points_std',
            'constructor_position_mean',
            'nationality_encoded',
            'nationality_constructor_encoded',
            'country_encoded'
        ]
        
        # Get win probabilities for each driver
        win_probs = self.model.predict_proba(pred_df[feature_columns])[:, 1]
        
        # Create results dataframe
        results = pd.DataFrame({
            'Driver': pred_df['driver_name'],
            'Team': pred_df['team_name'],
            'Grid': pred_df['grid'],
            'Win Probability': win_probs,
            'Championship Points': pred_df['points_championship']
        })
        
        return results.sort_values('Win Probability', ascending=False).reset_index(drop=True)
    
    def simulate_championship(self):
        """Simulate the remaining races of the 2025 championship."""
        if self.model is None or self.grid_2025 is None:
            return None
            
        # Initialize championship points with actual results from 2025
        championship_points = {driver: 0 for driver in self.grid_2025['driver_name']}
        
        # Add points from actual races
        if self.results_2025 is not None:
            for _, race in self.results_2025.iterrows():
                # F1 points system
                points = {
                    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
                    6: 8, 7: 6, 8: 4, 9: 2, 10: 1
                }
                if race['position'] in points:
                    championship_points[race['driver_name']] += points[race['position']]
                # Add point for fastest lap if applicable
                if pd.notna(race['fastest_lap']):
                    championship_points[race['driver_name']] += 1
        
        # Get remaining races
        completed_races = set()
        if self.results_2025 is not None:
            completed_races = set(self.results_2025['race_name'].unique())
        
        remaining_races = [
            race.split(" (")[0] for race in F1_CALENDAR_2025 
            if not race.endswith("COMPLETED")
        ]
        
        # Team reliability factors (1.0 = perfect reliability, higher = more problems)
        team_reliability = {
            'Red Bull Racing': 1.05,  # Top team with best reliability
            'Ferrari': 1.2,
            'Mercedes': 1.15,
            'McLaren': 1.2,
            'Aston Martin': 1.25,
            'Alpine': 1.3,
            'Williams': 1.35,
            'RB': 1.3,  # Ex AlphaTauri
            'Kick Sauber': 1.35,
            'Haas F1 Team': 1.4
        }
        
        # Driver error probability factors (higher = more prone to errors)
        driver_error_factor = {
            'Max Verstappen': 0.04,  # Campione in carica
            'Yuki Tsunoda': 0.08,    # Promosso in Red Bull
            'Charles Leclerc': 0.06,
            'Lewis Hamilton': 0.05,   # Esperienza in Ferrari
            'George Russell': 0.07,
            'Andrea Kimi Antonelli': 0.12,  # Rookie in Mercedes
            'Lando Norris': 0.06,
            'Oscar Piastri': 0.07,
            'Fernando Alonso': 0.05,  # Esperienza
            'Lance Stroll': 0.09,
            'Pierre Gasly': 0.08,
            'Jack Doohan': 0.11,      # Rookie
            'Alexander Albon': 0.08,
            'Carlos Sainz': 0.07,     # In Williams
            'Esteban Ocon': 0.08,
            'Oliver Bearman': 0.1,    # Rookie in Haas
            'Nico Hulkenberg': 0.08,
            'Gabriel Bortoleto': 0.12, # Rookie
            'Liam Lawson': 0.09,      # In RB
            'Isack Hadjar': 0.11      # Rookie in RB
        }
        
        # Simulate each remaining race
        race_results = []
        for race in remaining_races:
            # Predict race outcome
            results = self.predict_2025_race(race)
            if results is None:
                continue
            
            # Convert results to list for manipulation
            race_order = results.to_dict('records')
            
            # Simulate race incidents
            for i, driver in enumerate(race_order):
                team = driver['Team']
                driver_name = driver['Driver']
                
                # 1. DNF probability based on team reliability and track position
                base_dnf_prob = 0.02  # 2% base probability
                position_factor = 1 + (i * 0.01)  # Slightly higher chance of DNF for cars further back
                team_factor = team_reliability.get(team, 1.2)
                dnf_probability = base_dnf_prob * position_factor * team_factor
                
                if np.random.random() < dnf_probability:
                    race_order[i]['DNF'] = True
                    continue
                
                # 2. Driver errors (spins, missed braking points, etc.)
                error_prob = driver_error_factor.get(driver_name, 0.1)
                if np.random.random() < error_prob:
                    # Lose 2-5 positions
                    positions_lost = np.random.randint(2, 6)
                    new_pos = min(len(race_order) - 1, i + positions_lost)
                    # Swap positions
                    race_order.insert(new_pos, race_order.pop(i))
                
                # 3. Random penalties (5 or 10 seconds)
                if np.random.random() < 0.05:  # 5% chance of penalty
                    penalty_time = np.random.choice([5, 10])
                    # Simulate penalty effect on position
                    positions_lost = np.random.randint(1, 4)
                    new_pos = min(len(race_order) - 1, i + positions_lost)
                    race_order.insert(new_pos, race_order.pop(i))
                
                # 4. Pit stop issues
                if np.random.random() < 0.08:  # 8% chance of slow pit stop
                    # Lose 1-3 positions
                    positions_lost = np.random.randint(1, 4)
                    new_pos = min(len(race_order) - 1, i + positions_lost)
                    race_order.insert(new_pos, race_order.pop(i))
            
            # 5. Safety Car periods (30% chance per race)
            if np.random.random() < 0.3:
                # Safety Car bunches up the field and can lead to position swaps
                # Randomly swap some positions in the top 10
                for _ in range(np.random.randint(1, 4)):
                    pos1, pos2 = np.random.randint(0, min(10, len(race_order)), size=2)
                    race_order[pos1], race_order[pos2] = race_order[pos2], race_order[pos1]
            
            # Calculate points and update championship
            for pos, driver in enumerate(race_order):
                if driver.get('DNF', False):
                    continue
                    
                points = 0
                if pos < 10:  # Points positions
                    points_system = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
                    points = points_system[pos]
                    
                    # Fastest lap point (more likely for top 5, but possible for others in top 10)
                    fastest_lap_prob = 0.3 if pos < 5 else 0.1
                    if np.random.random() < fastest_lap_prob:
                        points += 1
                
                championship_points[driver['Driver']] += points
            
            # Store race results (only non-DNF drivers)
            valid_results = [d for d in race_order if not d.get('DNF', False)]
            if len(valid_results) >= 3:
                race_results.append({
                    'Race': race,
                    'Winner': valid_results[0]['Driver'],
                    'Second': valid_results[1]['Driver'],
                    'Third': valid_results[2]['Driver']
                })
        
        # Create final championship standings
        final_standings = pd.DataFrame({
            'Driver': list(championship_points.keys()),
            'Points': list(championship_points.values())
        })
        
        # Add team information and sort by points
        final_standings = pd.merge(
            final_standings,
            self.grid_2025[['driver_name', 'team_name']],
            left_on='Driver',
            right_on='driver_name'
        ).drop('driver_name', axis=1)
        
        final_standings = final_standings.sort_values('Points', ascending=False).reset_index(drop=True)
        
        # Calculate constructor standings
        constructor_standings = final_standings.groupby('team_name')['Points'].sum().reset_index()
        constructor_standings = constructor_standings.sort_values('Points', ascending=False).reset_index(drop=True)
        
        return final_standings, constructor_standings, race_results
    
    def train_model(self):
        # Get prepared features from data loader
        X_train, y_train, X_val, y_val, X_test, y_test = self.data_loader.prepare_features()
        
        # Initialize and train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Evaluate model
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        test_pred = self.model.predict(X_test)
        
        # Calculate probabilities for ROC AUC
        train_proba = self.model.predict_proba(X_train)[:, 1]
        val_proba = self.model.predict_proba(X_val)[:, 1]
        test_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'Train Accuracy': accuracy_score(y_train, train_pred),
            'Validation Accuracy': accuracy_score(y_val, val_pred),
            'Test Accuracy': accuracy_score(y_test, test_pred),
            'Train ROC AUC': roc_auc_score(y_train, train_proba),
            'Validation ROC AUC': roc_auc_score(y_val, val_proba),
            'Test ROC AUC': roc_auc_score(y_test, test_proba)
        }
        
        # Calculate precision, recall, and F1 score for test set
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_pred, average='binary')
        metrics.update({
            'Test Precision': precision,
            'Test Recall': recall,
            'Test F1 Score': f1
        })
        
        return metrics
    
    def save_model(self, filename='f1_model.joblib'):
        if self.model is not None:
            model_data = {
                'model': self.model,
                'feature_importance': self.feature_importance,
                'label_encoders': self.data_loader.label_encoders
            }
            joblib.dump(model_data, filename)
            return f"Model saved to {filename}"
    
    def load_model(self, filename='f1_model.joblib'):
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.feature_importance = model_data['feature_importance']
        self.data_loader.label_encoders = model_data['label_encoders']
        return f"Model loaded from {filename}"

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def main():
    # Load animations
    lottie_race = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_0yfsb3a1.json")
    lottie_trophy = load_lottieurl("https://lottie.host/3ed0f744-3c8c-42bd-a471-b17eaab57ffb/daCVHt1yDp.json")
    lottie_developer = load_lottieurl("https://lottie.host/acb26878-f6c7-4fa3-bcfe-d7c11d9f5ccc/RkpBW9bdFE.json")
    lottie_education = load_lottieurl("https://lottie.host/ea45341b-3846-413c-ae38-611449305ef8/Eh9NrUQ8WD.json")
    lottie_skills = load_lottieurl("https://lottie.host/3033fbf9-5575-4abf-b342-38f1282cacd1/7GRHwbkBNR.json")
    lottie_experience = load_lottieurl("https://lottie.host/1c1cfd36-21d1-4ec7-b15e-7d0169815c68/gRFKd8spZ8.json")
    lottie_projects = load_lottieurl("https://lottie.host/bd40a302-53cc-4685-88fb-293406119a4a/wXyjii2gRt.json")
    lottie_honors = load_lottieurl("https://lottie.host/247d4b3a-1b2e-4deb-9611-6de44f0bfc23/5x96E8csNO.json")

    # Custom CSS
    st.markdown(f"""
    <style>
        .main {{
            background-color: #0E1117;
            color: #FAFAFA;
        }}
        .stApp {{
            background-image: linear-gradient(to bottom, #000000, #1E1E1E);
            color: white;
        }}
        .header-container {{
            display: flex;
            align-items: center;
            margin-bottom: -30px;
        }}
        .header-title {{
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(to right, #FF1801, #FF8C00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        .header-subtitle {{
            font-size: 1.2rem;
            color: #AAAAAA;
            margin-bottom: 2rem;
        }}
        .profile-card {{
            background-color: #1E1E1E;
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
            border-left: 5px solid #FF1801;
            transition: transform 0.3s;
        }}
        .profile-card:hover {{
            transform: translateY(-5px);
        }}
        .section-title {{
            font-size: 1.8rem;
            color: #FF1801;
            margin-top: 30px;
            margin-bottom: 15px;
            border-bottom: 2px solid #FF1801;
            padding-bottom: 5px;
        }}
        .experience-card {{
            background-color: #1E1E1E;
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            border-left: 4px solid #00D2BE;
        }}
        .project-card {{
            background-color: #1E1E1E;
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            border-left: 4px solid #0090FF;
        }}
        .education-card {{
            background-color: #1E1E1E;
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            border-left: 4px solid #FF8700;
        }}
        .honors-card {{
            background-color: #1E1E1E;
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            border-left: 4px solid #52E252;
        }}
        .skill-pill {{
            display: inline-block;
            background-color: #333333;
            color: white;
            padding: 5px 15px;
            margin: 5px;
            border-radius: 20px;
            font-size: 0.9rem;
        }}
        .tab-content {{
            animation: fadeIn 0.5s ease-in-out;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .stButton>button {{
            background-color: #FF1801;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 24px;
            font-weight: bold;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: #CC0000;
            transform: scale(1.05);
        }}
        .stSelectbox>div>div>select {{
            background-color: #333333;
            color: white;
        }}
        .stNumberInput>div>div>input {{
            background-color: #333333;
            color: white;
        }}
        .stCheckbox>label {{
            color: white;
        }}
        .css-1aumxhk {{
            background-color: #1E1E1E;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .timeline {{
            position: relative;
            max-width: 1200px;
            margin: 0 auto;
        }}
        .timeline::after {{
            content: '';
            position: absolute;
            width: 6px;
            background-color: #FF1801;
            top: 0;
            bottom: 0;
            left: 50%;
            margin-left: -3px;
        }}
        .container {{
            padding: 10px 40px;
            position: relative;
            background-color: inherit;
            width: 50%;
        }}
        .container::after {{
            content: '';
            position: absolute;
            width: 25px;
            height: 25px;
            right: -17px;
            background-color: #FF1801;
            border: 4px solid #FF1801;
            top: 15px;
            border-radius: 50%;
            z-index: 1;
        }}
        .left {{
            left: 0;
        }}
        .right {{
            left: 50%;
        }}
        .left::before {{
            content: " ";
            height: 0;
            position: absolute;
            top: 22px;
            width: 0;
            z-index: 1;
            right: 30px;
            border: medium solid #FF1801;
            border-width: 10px 0 10px 10px;
            border-color: transparent transparent transparent #FF1801;
        }}
        .right::before {{
            content: " ";
            height: 0;
            position: absolute;
            top: 22px;
            width: 0;
            z-index: 1;
            left: 30px;
            border: medium solid #FF1801;
            border-width: 10px 10px 10px 0;
            border-color: transparent #FF1801 transparent transparent;
        }}
        .right::after {{
            left: -16px;
        }}
        .content {{
            padding: 20px 30px;
            background-color: #1E1E1E;
            position: relative;
            border-radius: 6px;
        }}
        .social-icon {{
            font-size: 24px;
            margin: 0 10px;
            color: white;
            transition: all 0.3s ease;
        }}
        .social-icon:hover {{
            color: #FF1801;
            transform: scale(1.2);
        }}
        .creative-card {{
            background-color: #1E1E1E;
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            border-left: 4px solid #DC0000;
        }}
    </style>
    """, unsafe_allow_html=True)

    st.set_page_config(
        page_title="F1 ORACLE: 2025 F1 Race Predictor by Shashwat Kashyap",
        page_icon="🏎️",
        layout="wide"
    )

    # Main header with logo
    st.markdown("""
    <div class="header-container">
        <div style="flex: 1;">
            <div class="header-title">F1 ORACLE: Formula 1 2025 Race Predictor</div>
            <div class="header-subtitle">AI-powered predictions for the 2025 Formula 1 season</div>
        </div>
        <div style="margin-left: 20px;">
            <img src="https://www.formula1.com/etc/designs/fom-website/images/f1_logo.svg" width="120">
        </div>
    </div>
    """, unsafe_allow_html=True)

    predictor = F1Predictor()

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏁 Race Predictions", "🏆 Championship", "🛠 Model Training", "👨‍💻 About Me"])

    with tab1:
        if predictor.model is None:
            st.warning("⚠️ Please train a model first or load an existing model.")
            if st.button("🔌 Load Existing Model", key="load_model_race"):
                try:
                    with st.spinner("Loading model..."):
                        load_message = predictor.load_model()
                        st.success(load_message)
                        st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
        else:
            st.markdown("### 2025 Race Predictions")
            
            # Add circuit selection from calendar
            circuit = st.selectbox(
                "🏟️ Select Circuit",
                F1_CALENDAR_2025,
                key="circuit_select"
            )
            
            # Option to modify grid positions
            modify_grid = st.checkbox("✏️ Customize Grid Positions", key="modify_grid")
            
            qualifying_results = None
            if modify_grid:
                st.markdown("### 🏁 Set Grid Positions")
                st.write("Adjust the starting positions for each driver:")
                
                cols = st.columns(4)
                qualifying_results = {}
                team_colors = TEAM_COLORS
                
                for idx, row in predictor.grid_2025.iterrows():
                    with cols[idx % 4]:
                        team_color = team_colors.get(row['team_name'], '#333333')
                        st.markdown(f"""
                            <div style="background-color: #1E1E1E; border-radius: 10px; padding: 15px; margin: 10px 0; border-left: 4px solid {team_color};">
                                <h4>{row['driver_name']}</h4>
                                <p><small>{row['team_name']}</small></p>
                            </div>
                        """, unsafe_allow_html=True)
                        pos = st.number_input(
                            f"Position for {row['driver_name']}",
                            min_value=1,
                            max_value=20,
                            value=idx + 1,
                            key=f"grid_pos_{row['driverId']}"
                        )
                        qualifying_results[row['driverId']] = pos
            
            if st.button("🔮 Predict Race Outcome", key="predict_race"):
                with st.spinner("Calculating predictions..."):
                    results = predictor.predict_2025_race(circuit, qualifying_results)
                    
                    if results is not None:
                        st.markdown(f"### 🏁 Predicted Race Results for {circuit.split(' - ')[0]}")
                        
                        # Create animated podium visualization
                        if len(results) >= 3:
                            podium_fig = go.Figure()
                            
                            # Podium steps
                            podium_fig.add_trace(go.Bar(
                                x=["🥇 2nd", "🥈 1st", "🥉 3rd"],
                                y=[2, 3, 1],
                                marker_color=['#CD7F32', '#C0C0C0', '#FFD700'],
                                hoverinfo='none',
                                showlegend=False
                            ))
                            
                            # Driver names on podium
                            podium_fig.add_annotation(
                                x="🥈 1st", y=3.2,
                                text=results.iloc[0]['Driver'],
                                showarrow=False,
                                font=dict(size=14, color='white')
                            )
                            podium_fig.add_annotation(
                                x="🥇 2nd", y=2.2,
                                text=results.iloc[1]['Driver'],
                                showarrow=False,
                                font=dict(size=14, color='white')
                            )
                            podium_fig.add_annotation(
                                x="🥉 3rd", y=1.2,
                                text=results.iloc[2]['Driver'],
                                showarrow=False,
                                font=dict(size=14, color='white')
                            )
                            
                            podium_fig.update_layout(
                                height=400,
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                xaxis=dict(showgrid=False, zeroline=False),
                                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                                margin=dict(l=0, r=0, t=0, b=0),
                                transition={'duration': 1000}
                            )
                            
                            st.plotly_chart(podium_fig, use_container_width=True)
                        
                        # Create results table with team colors
                        results['Team Color'] = results['Team'].map(TEAM_COLORS)
                        
                        fig = go.Figure(data=[
                            go.Table(
                                columnwidth=[50, 150, 150, 50, 100, 100],
                                header=dict(
                                    values=['Pos', 'Driver', 'Team', 'Grid', 'Win %', 'Points'],
                                    fill_color='#FF1801',
                                    align=['center', 'left', 'left', 'center', 'center', 'center'],
                                    font=dict(color='white', size=12),
                                    height=40
                                ),
                                cells=dict(
                                    values=[
                                        list(range(1, len(results) + 1)),
                                        results['Driver'],
                                        results['Team'],
                                        results['Grid'],
                                        [f"{x:.1%}" for x in results['Win Probability']],
                                        results['Championship Points']
                                    ],
                                    fill_color=['#1E1E1E', '#1E1E1E', results['Team Color'], '#1E1E1E', '#1E1E1E', '#1E1E1E'],
                                    align=['center', 'left', 'left', 'center', 'center', 'center'],
                                    font=dict(color='white', size=11),
                                    height=30
                                )
                            )
                        ])
                        
                        fig.update_layout(
                            margin=dict(l=0, r=0, t=0, b=0),
                            height=min(800, 35 * len(results) + 50),
                            transition={'duration': 500}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add a bar chart of win probabilities with team colors
                        top_drivers = results.head(10).copy()
                        top_drivers['Team Color'] = top_drivers['Team'].map(TEAM_COLORS)
                        
                        prob_fig = px.bar(
                            top_drivers,
                            x='Driver',
                            y='Win Probability',
                            title='',
                            color='Team',
                            color_discrete_map=TEAM_COLORS,
                            text=[f"{x:.1%}" for x in top_drivers['Win Probability']],
                            hover_data=['Team']
                        )
                        
                        prob_fig.update_traces(
                            textposition='outside',
                            marker_line_width=0,
                            hovertemplate="<b>%{x}</b><br>Team: %{customdata[0]}<br>Win Probability: %{y:.1%}<extra></extra>"
                        )
                        
                        prob_fig.update_layout(
                            height=500,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            xaxis_title="",
                            yaxis_title="Win Probability",
                            yaxis_tickformat=".0%",
                            transition={'duration': 500},
                            hoverlabel=dict(
                                bgcolor="#1E1E1E",
                                font_size=12,
                                font_family="Arial"
                            )
                        )
                        
                        st.plotly_chart(prob_fig, use_container_width=True)

    with tab2:
        if predictor.model is None:
            st.warning("⚠️ Please train a model first or load an existing model.")
            if st.button("🔌 Load Existing Model", key="load_model_championship"):
                try:
                    with st.spinner("Loading model..."):
                        load_message = predictor.load_model()
                        st.success(load_message)
                        st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
        else:
            st.markdown("### 2025 Championship Simulation")
            st.write("Simulate the remaining races of the 2025 season with realistic race incidents and outcomes.")
            
            if lottie_trophy:
                st_lottie(lottie_trophy, height=200, key="trophy")
            
            if st.button("🎮 Simulate Championship", key="simulate_championship"):
                with st.spinner("Running championship simulation... This may take a minute"):
                    progress_bar = st.progress(0)
                    
                    # Simulate with progress updates
                    def simulate_with_progress():
                        for i in range(10):
                            time.sleep(0.1)
                            progress_bar.progress((i + 1) * 10)
                        return predictor.simulate_championship()
                    
                    driver_standings, constructor_standings, race_results = simulate_with_progress()
                    
                    st.success("Championship simulation completed!")
                    
                    # Display Driver's Championship
                    st.markdown("### 🏆 Predicted Driver's Championship Standings")
                    
                    # Add team colors
                    driver_standings['Team Color'] = driver_standings['team_name'].map(TEAM_COLORS)
                    
                    # Create a podium visualization for top 3
                    if len(driver_standings) >= 3:
                        podium_fig = go.Figure()
                        
                        # Podium steps
                        podium_fig.add_trace(go.Bar(
                            x=["🥇 2nd", "🥈 1st", "🥉 3rd"],
                            y=[driver_standings.iloc[1]['Points'], 
                            driver_standings.iloc[0]['Points'], 
                            driver_standings.iloc[2]['Points']],
                            marker_color=['#CD7F32', '#C0C0C0', '#FFD700'],
                            hoverinfo='none',
                            showlegend=False
                        ))
                        
                        # Driver names on podium
                        podium_fig.add_annotation(
                            x="🥈 1st", y=driver_standings.iloc[0]['Points'] + 10,
                            text=f"{driver_standings.iloc[0]['Driver']}<br>{driver_standings.iloc[0]['Points']} pts",
                            showarrow=False,
                            font=dict(size=14, color='white')
                        )
                        podium_fig.add_annotation(
                            x="🥇 2nd", y=driver_standings.iloc[1]['Points'] + 10,
                            text=f"{driver_standings.iloc[1]['Driver']}<br>{driver_standings.iloc[1]['Points']} pts",
                            showarrow=False,
                            font=dict(size=14, color='white')
                        )
                        podium_fig.add_annotation(
                            x="🥉 3rd", y=driver_standings.iloc[2]['Points'] + 10,
                            text=f"{driver_standings.iloc[2]['Driver']}<br>{driver_standings.iloc[2]['Points']} pts",
                            showarrow=False,
                            font=dict(size=14, color='white')
                        )
                        
                        podium_fig.update_layout(
                            height=400,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(showgrid=False, zeroline=False),
                            yaxis=dict(showgrid=False, zeroline=False, visible=False),
                            margin=dict(l=0, r=0, t=0, b=0),
                            transition={'duration': 1000}
                        )
                        
                        st.plotly_chart(podium_fig, use_container_width=True)
                    
                    # Full standings table
                    fig_drivers = go.Figure(data=[
                        go.Table(
                            columnwidth=[50, 200, 150, 100],
                            header=dict(
                                values=['Pos', 'Driver', 'Team', 'Points'],
                                fill_color='#FF1801',
                                align=['center', 'left', 'left', 'center'],
                                font=dict(color='white', size=12),
                                height=40
                            ),
                            cells=dict(
                                values=[
                                    list(range(1, len(driver_standings) + 1)),
                                    driver_standings['Driver'],
                                    driver_standings['team_name'],
                                    driver_standings['Points']
                                ],
                                fill_color=['#1E1E1E', '#1E1E1E', driver_standings['Team Color'], '#1E1E1E'],
                                align=['center', 'left', 'left', 'center'],
                                font=dict(color='white', size=11),
                                height=30
                            )
                        )
                    ])
                    fig_drivers.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0), 
                        height=min(800, 35 * len(driver_standings) + 50),
                        transition={'duration': 500}
                    )
                    st.plotly_chart(fig_drivers, use_container_width=True)
                    
                    # Constructor standings
                    st.markdown("### 🏭 Predicted Constructor's Championship Standings")
                    
                    constructor_standings['Team Color'] = constructor_standings['team_name'].map(TEAM_COLORS)
                    
                    # Bar chart visualization
                    constructor_fig = px.bar(
                        constructor_standings,
                        x='team_name',
                        y='Points',
                        color='team_name',
                        color_discrete_map=TEAM_COLORS,
                        text='Points',
                        title=''
                    )
                    
                    constructor_fig.update_traces(
                        textposition='outside',
                        marker_line_width=0,
                        hovertemplate="<b>%{x}</b><br>Points: %{y}<extra></extra>"
                    )
                    
                    constructor_fig.update_layout(
                        height=500,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        xaxis_title="",
                        yaxis_title="Points",
                        showlegend=False,
                        transition={'duration': 500}
                    )
                    
                    st.plotly_chart(constructor_fig, use_container_width=True)
                    
                    # Race results
                    st.markdown("### 🏁 Predicted Race Winners")
                    
                    # Create DataFrame from race_results
                    race_results_df = pd.DataFrame(race_results)
                    race_results_df['Race Number'] = range(1, len(race_results_df) + 1)
                    
                    # Timeline visualization
                    timeline_fig = px.timeline(
                        race_results_df,
                        x_start="Race Number",
                        x_end="Race Number",
                        y="Race",
                        color_discrete_sequence=['#FF1801'],
                        hover_name="Race",
                        hover_data=["Winner", "Second", "Third"],
                        title=""
                    )
                    
                    timeline_fig.update_yaxes(autorange="reversed")
                    timeline_fig.update_traces(
                        marker=dict(
                            line=dict(width=2, color='white')  # Add a white border to markers
                        ),
                        hovertemplate="<b>%{hovertext}</b><br>🥇 %{customdata[0]}<br>🥈 %{customdata[1]}<br>🥉 %{customdata[2]}<extra></extra>"
                    )
                    
                    timeline_fig.update_layout(
                        height=800,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        xaxis_title="Race Number",
                        yaxis_title="",
                        transition={'duration': 500}
                    )
                    
                    st.plotly_chart(timeline_fig, use_container_width=True)
                    
                    # Points progression visualization
                    st.markdown("### 📈 Championship Points Progression")
                    
                    # This would require tracking points through the season, which would need additional implementation
                    # For now, we'll show the final points distribution
                    points_fig = px.bar(
                        driver_standings.head(10),
                        x='Driver',
                        y='Points',
                        color='team_name',
                        color_discrete_map=TEAM_COLORS,
                        text='Points',
                        title=''
                    )
                    
                    points_fig.update_traces(
                        textposition='outside',
                        marker_line_width=0,
                        hovertemplate="<b>%{x}</b><br>Team: %{customdata[0]}<br>Points: %{y}<extra></extra>"
                    )
                    
                    points_fig.update_layout(
                        height=500,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        xaxis_title="",
                        yaxis_title="Points",
                        showlegend=False,
                        transition={'duration': 500}
                    )
                    
                    st.plotly_chart(points_fig, use_container_width=True)

    with tab3:
        if st.button('🚀 Train New Model', key='train_button'):
            with st.spinner('Training model... This may take a few minutes'):
                try:
                    progress_bar = st.progress(0)
                    for percent_complete in range(100):
                        time.sleep(0.02)  # Simulate progress
                        progress_bar.progress(percent_complete + 1)
                    
                    metrics = predictor.train_model()
                    
                    st.success('Model trained successfully!')
                    
                    # Display metrics in cards
                    st.markdown("### Model Performance Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown('<div class="profile-card">'
                                   '<h3>Accuracy Metrics</h3>'
                                   f'<p>Train: <strong>{metrics["Train Accuracy"]:.2%}</strong></p>'
                                   f'<p>Validation: <strong>{metrics["Validation Accuracy"]:.2%}</strong></p>'
                                   f'<p>Test: <strong>{metrics["Test Accuracy"]:.2%}</strong></p>'
                                   '</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="profile-card">'
                                   '<h3>ROC AUC Scores</h3>'
                                   f'<p>Train: <strong>{metrics["Train ROC AUC"]:.2%}</strong></p>'
                                   f'<p>Validation: <strong>{metrics["Validation ROC AUC"]:.2%}</strong></p>'
                                   f'<p>Test: <strong>{metrics["Test ROC AUC"]:.2%}</strong></p>'
                                   '</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="profile-card">'
                                   '<h3>Test Set Metrics</h3>'
                                   f'<p>Precision: <strong>{metrics["Test Precision"]:.2%}</strong></p>'
                                   f'<p>Recall: <strong>{metrics["Test Recall"]:.2%}</strong></p>'
                                   f'<p>F1 Score: <strong>{metrics["Test F1 Score"]:.2%}</strong></p>'
                                   '</div>', unsafe_allow_html=True)
                    
                    # Feature importance plot with animation
                    st.markdown("### Feature Importance")
                    fig = px.bar(
                        predictor.feature_importance,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title='',
                        color='importance',
                        color_continuous_scale='reds'
                    )
                    fig.update_layout(
                        xaxis_title='Importance Score',
                        yaxis_title='Feature',
                        height=600,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        transition={'duration': 500}
                    )
                    fig.update_traces(marker_line_width=0)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save the model
                    save_message = predictor.save_model()
                    st.info(save_message)
                    
                except Exception as e:
                    st.error(f'Error during model training: {str(e)}')

    with tab4:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="color: #FF1801; font-size: 3rem;">SHASHWAT KASHYAP</h1>
            <p style="font-size: 1.5rem; color: #AAAAAA;">Full-Stack Developer & Data Scientist</p>
            <div style="margin-top: 20px;">
                <a href="https://www.linkedin.com/in/shashwat-kashyap-b417aa155" class="social-icon" target="_blank">🔗</a>
                <a href="mailto:kashyapshashwat77@gmail.com" class="social-icon" target="_blank">✉️</a>
                <a href="tel:+919818383499" class="social-icon" target="_blank">📞</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if lottie_developer:
            st_lottie(lottie_developer, height=300, key="developer")

        st.markdown('<div class="section-title">Professional Summary</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="profile-card">
            <p>Full-stack Software Developer (MERN, Python, Java) with a proven record of building scalable, 
            production-ready applications that deliver impact across education, health, and enterprise domains. 
            Adept at transforming abstract ideas into polished products, optimizing workflows (up to 40%), 
            and leading cross-functional collaboration. Skilled in agile development, problem-solving, 
            and research-backed innovation. Actively pursuing excellence at the intersection of technology, 
            human behaviour, and design.</p>
        </div>
        """, unsafe_allow_html=True)
        if lottie_skills:
                st_lottie(lottie_skills, height=200, key="skills")

        st.markdown('<div class="section-title">Technical Skills</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-bottom: 30px;">
            <h4>Languages</h4>
            <span class="skill-pill">Python</span>
            <span class="skill-pill">Java</span>
            <span class="skill-pill">JavaScript</span>
        </div>
        <div style="margin-bottom: 30px;"> 
            <h4>Frameworks & Libraries</h4>
            <span class="skill-pill">React.js</span>
            <span class="skill-pill">Node.js</span>
            <span class="skill-pill">Express.js</span>
            <span class="skill-pill">MongoDB</span>
            <span class="skill-pill">MERN Stack</span>
        </div>
        <div style="margin-bottom: 30px;"> 
            <h4>Databases</h4>
            <span class="skill-pill">MySQL</span>
            <span class="skill-pill">MongoDB</span>
        </div>
        <div style="margin-bottom: 30px;">   
            <h4>Tools & Platforms</h4>
            <span class="skill-pill">Git</span>
            <span class="skill-pill">GitHub</span>
            <span class="skill-pill">VS Code</span>
            <span class="skill-pill">Figma</span>
            <span class="skill-pill">Canva</span>
            <span class="skill-pill">Render</span>
        </div>
        <div style="margin-bottom: 30px;">
            <h4>Other</h4>
            <span class="skill-pill">REST APIs</span>
            <span class="skill-pill">SEO Optimization</span>
            <span class="skill-pill">Agile</span>
            <span class="skill-pill">Sentiment Analysis</span>
            <span class="skill-pill">EDA</span>
            <span class="skill-pill">Technical Documentation</span>
        </div>
        """, unsafe_allow_html=True)


        st.markdown('<div class="section-title">Education</div>', unsafe_allow_html=True)
        if lottie_education:
            st_lottie(lottie_education, height=200, key="education")
        col_edu1, col_edu2 = st.columns(2)
        with col_edu1:
            st.markdown("""
            <div class="education-card">
                <h4>Master of Computer Applications (MCA)</h4>
                <p><strong>Birla Institute of Technology, Mesra</strong></p>
                <p>2024 - 2026 | CGPA: 7.71</p>
            </div>
            """, unsafe_allow_html=True)
        with col_edu2:
            st.markdown("""
            <div class="education-card">
                <h4>Bachelor of Computer Applications (BCA)</h4>
                <p><strong>Birla Institute of Technology, Mesra</strong></p>
                <p>2021 - 2024 | CGPA: 7.78</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Professional Experience</div>', unsafe_allow_html=True)
        if lottie_experience:
            st_lottie(lottie_experience, height=200, key="experience")
        
        # Experience Timeline
        st.markdown("""
        <div class="timeline">
            <div class="container left">
                <div class="content">
                    <h4>Full Stack Developer Intern</h4>
                    <p><strong>Pinaqtech | May 2025 - Present</strong></p>
                    <ul>
                        <li>Developed and deployed the company's official website using MERN stack; boosted SEO ranking by 40%</li>
                        <li>Integrated security features and optimized user experience across platforms</li>
                    </ul>
                </div>
            </div>
            <div class="container right">
                <div class="content">
                    <h4>Software Engineering Intern</h4>
                    <p><strong>Oohr Innovations | May 2025 - Present</strong></p>
                    <ul>
                        <li>Built a custom ERP system for educational institutions; automated academic workflows, reducing manual tasks by 60%</li>
                        <li>Led QA testing, resolved 20+ critical bugs, and enhanced system reliability</li>
                    </ul>
                </div>
            </div>
            <div class="container left">
                <div class="content">
                    <h4>MIS Intern</h4>
                    <p><strong>Furnish Me Architects | May 2025 - Present</strong></p>
                    <ul>
                        <li>Engineered a secure, centralized intranet for cross-team reporting; improved access and reporting speed by 40%</li>
                    </ul>
                </div>
            </div>
            <div class="container right">
                <div class="content">
                    <h4>Senior Developer Intern - Government Project</h4>
                    <p><strong>DST Menstrual Hygiene App | Jul 2023 - Apr 2024</strong></p>
                    <ul>
                        <li>Developed scalable features for a public health app serving 5,000+ users</li>
                        <li>Collaborated with health professionals to translate medical insights into technical solutions</li>
                    </ul>
                </div>
            </div>
            <div class="container left">
                <div class="content">
                    <h4>Python Developer Intern</h4>
                    <p><strong>CodSoft | September 2023 - October 2023</strong></p>
                    <ul>
                        <li>Built 5 automation tools for data handling, sentiment analysis, and task scheduling</li>
                    </ul>
                </div>
            </div>
            <div class="container right">
                <div class="content">
                    <h4>Operations Intern - ICIEM Conference</h4>
                    <p><strong>BIT Mesra | May 2023 - June 2023</strong></p>
                    <ul>
                        <li>Managed logistics for 100+ international delegates</li>
                        <li>Reduced document turnaround time by 40% through workflow automation</li>
                    </ul>
                </div>
            </div>
            <div class="container left">
                <div class="content">
                    <h4>Marketing Intern - BIT Admission Campaign</h4>
                    <p><strong>BIT Mesra | May 2023 - July 2023</strong></p>
                    <ul>
                        <li>Designed and A/B tested marketing creatives, increasing applications by 20%</li>
                    </ul>
                </div>
            </div>
            <div class="container right">
                <div class="content">
                    <h4>Senior Intern - BIT Noida</h4>
                    <p><strong>BIT Mesra | May 2024 - July 2024</strong></p>
                    <ul>
                        <li>Led data-driven social media strategies; improved admission inquiries by 30%</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Projects</div>', unsafe_allow_html=True)
        if lottie_projects:
            st_lottie(lottie_projects, height=200, key="projects")
        col_proj1, col_proj2 = st.columns(2)
        with col_proj1:
            st.markdown("""
            <div class="project-card">
                <h4>GradLink — Alumni Networking Portal</h4>
                <p><em>MERN Stack | Render Deployment | 2023</em></p>
                <ul>
                    <li>Built a responsive platform connecting 5,000+ BIT Mesra alumni and students</li>
                    <li>Implemented user authentication, chat features, and profile discovery</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with col_proj2:
            st.markdown("""
            <div class="project-card">
                <h4>Alfred — AI-Powered Voice Assistant</h4>
                <p><em>Python | 2023</em></p>
                <ul>
                    <li>Designed and deployed a voice-controlled assistant for desktop automation with 90% command accuracy</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        col_proj3, _ = st.columns(2)
        with col_proj3:
            st.markdown("""
            <div class="project-card">
                <h4>SkyVault — Secure File Sharing Platform</h4>
                <p><em>MERN Stack | 2024</em></p>
                <ul>
                    <li>Created an encrypted cloud-based sharing tool using access-control logic and role-based permissions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Honors & Publications</div>', unsafe_allow_html=True)
        if lottie_honors:
            st_lottie(lottie_honors, height=200, key="honors")
        col_honors1, col_honors2 = st.columns(2)
        with col_honors1:
            st.markdown("""
            <div class="honors-card">
                <h4>Scholarships</h4>
                <ul>
                    <li>GP Birla Foundation Scholarship</li>
                    <li>Dr. Usha Aggarwal Trust Scholarship</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with col_honors2:
            st.markdown("""
            <div class="honors-card">
                <h4>Published Articles</h4>
                <ul>
                    <li>"The Potential Impact of Quantum Computing, AI & Unified Theory on our Understanding of Time" — Bits & Bytes</li>
                    <li>"The Age of the Forgotten Mind" — BIT Newsletter</li>
                    <li>"A Bittersweet Farewell" — WhyNot Australia</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Leadership & Extracurriculars</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="profile-card">
            <ul>
                <li><strong>Coordinator — IT Club:</strong> Conducted workshops on Git, MERN, and web deployment; mentored 50+ juniors</li>
                <li><strong>Coordinator — Literature Society:</strong> Hosted debates, literary events, and open mics</li>
                <li><strong>Coordinator — Sports Club (Badminton):</strong> Organized 6+ tournaments; trained new entrants</li>
                <li><strong>Member — News & Publication Society:</strong> Wrote and edited stories on innovation and tech culture</li>
                <li><strong>Core Team — Entrepreneurship Club:</strong> Promoted internship drives and startup incubations</li>
                <li><strong>Cultural Club Volunteer:</strong> Managed fests, stage shows, and inter-college cultural exchanges</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


        st.markdown('<div class="section-title">Creative Works</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="creative-card">
            <h4>Poetry</h4>
            <ul>
                <li><em>Alone</em> (BIT Fusion)</li>
                <li><em>Symphony of Despair</em> (WhyNot Australia)</li>
                <li><em>Echoes in the Abyss</em> (WhyNot Australia)</li>
                <li><em>The Greyness of Man</em> (WhyNot Australia)</li>
            </ul>
        </div>
    
        <div class="creative-card">
            <h4>Anthologies</h4>
            <ul>
                <li><em>Be Limitless</em> (World Book Fair 2025)</li>
                <li><em>Chromatic Currents V3</em></li>
                <li><em>Golden Realms V4</em></li>
            </ul>
        </div>
                    
        

        """, unsafe_allow_html=True)
    # 🧾 Footer
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-style: italic;'>"
        "Built during a caffeine overdose, under the influence of sleep deprivation, reckless ambition, "
        "and just enough genius to make it work — which, frankly, is still more coherent than most Ferrari strategy decisions.<br>"
        "— Shashwat Kashyap ☕💻🏎️"
        "</p>",
        unsafe_allow_html=True
    )



if __name__ == "__main__":
    main()