import os
import fastf1
import pandas as pd
import numpy as np
from fastf1.ergast import Ergast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib

cache_dir = 'f1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)

fastf1.Cache.enable_cache(cache_dir)
ergast = Ergast()

all_race_rows = []
seasons_to_extract = [2024, 2025]

print("Extracting 2024 and 2025 season data...")

for season in seasons_to_extract:
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    
    for index, event in schedule.iterrows():
        event_name = event['EventName']
        round_num = event['RoundNumber']
        
        # Skip Round 1 of every season to avoid the "0 points" cold-start problem
        if round_num <= 1:
            continue
            
        print(f"Processing {season} Round {round_num}: {event_name}...")
        
        try:
            q_session = fastf1.get_session(season, round_num, 'Q')
            r_session = fastf1.get_session(season, round_num, 'R')
            
            q_session.load(telemetry=False, weather=False, messages=False)
            r_session.load(telemetry=False, weather=False, messages=False)
            
            q_laps = q_session.laps.pick_accurate()
            if q_laps.empty:
                continue
                
            fastest_q_laps = q_laps.groupby('Driver')['LapTime'].min().dt.total_seconds()
            if fastest_q_laps.empty:
                continue
                
            pole_time = fastest_q_laps.min()
            
            # Fetch Ergast standings for the specific season and previous round
            team_ranks = {}
            team_points_dict = {}
            
            try:
                standings_resp = ergast.get_constructor_standings(season=season, round=round_num - 1)
                if standings_resp.content:
                    c_df = standings_resp.content[0]
                    team_ranks = dict(zip(c_df['constructorName'], c_df['position']))
                    team_points_dict = dict(zip(c_df['constructorName'], c_df['points']))
            except Exception as e:
                print(f"  Warning: Ergast error on {season} round {round_num}: {e}")

            results = r_session.results
                
            for driver in results['Abbreviation']:
                driver_res = results[results['Abbreviation'] == driver]
                if driver_res.empty:
                    continue
                    
                driver_res = driver_res.iloc[0]
                status = str(driver_res['Status']).lower()
                classified_pos = driver_res['ClassifiedPosition']
                    
                try:
                    if 'finished' in status or '+' in status:
                            pos = int(classified_pos)
                            if pos <= 3:
                                finish_bracket = 0      # Podium
                            elif pos <= 10:
                                finish_bracket = 1      # Points
                            else:
                                finish_bracket = 2      # Lower Bracket
                    else:
                            finish_bracket = 2
                except (ValueError, TypeError):
                            finish_bracket = 2
                
                if driver in fastest_q_laps and pd.notna(fastest_q_laps[driver]):
                    q_time = fastest_q_laps[driver]
                    quali_delta = max(0.0, float(q_time - pole_time))
                else:
                    quali_delta = 2.5
                    
                quali_pos = int(driver_res['GridPosition']) if pd.notna(driver_res['GridPosition']) else 20
                grid_start = int(driver_res['GridPosition']) if pd.notna(driver_res['GridPosition']) else 20
                grid_penalty = max(0, grid_start - quali_pos) if quali_pos > 0 else 0
                
                driver_team = driver_res['TeamName']
                constructor_pos = int(team_ranks.get(driver_team, 5))
                
                cum_points = float(team_points_dict.get(driver_team, 0.0))
                team_ppg = cum_points / float(round_num - 1)
                
                all_race_rows.append({
                    'QualiPosition': quali_pos,
                    'GridPenalty': grid_penalty,
                    'QualiPaceDelta': quali_delta,
                    'TeamPPG': team_ppg,
                    'ConstructorPos': constructor_pos,
                    'FinishBracket': finish_bracket
                })
                
        except Exception as e:
            print(f"Skipping {season} {event_name} due to error: {e}")
            continue

df = pd.DataFrame(all_race_rows)
print(f"\nSuccessfully extracted {len(df)} total driver race entries!")

X = df.drop(columns=['FinishBracket'])
y = df['FinishBracket']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

num_features = X_train_scaled.shape[1]

# 32 -> 16 architecture is perfect for ~700 rows of data
model = models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(num_features,)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train_scaled, 
    y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test_scaled, y_test),
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nFinal Test Set Accuracy: {test_acc * 100:.2f}%")

model.save('f1_predictor_model.keras')
joblib.dump(scaler, 'scaler.pkl')