[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![FastF1](https://img.shields.io/badge/Data-FastF1-E10600?style=for-the-badge&logo=formula1&logoColor=white)](https://docs.fastf1.dev/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)


# F1 Race Finish Predictor 🏎️ 🏁 

This Machine Learning model (a feedforward neural network) predicts a Formula 1 driver's race outcome—specifically whether they will secure a podium, finish in the points, or have a bad weekend. The predictions are based on qualifying pace, constructor performance, and relative grid positioning.

## How It Works

Instead of trying to predict highly chaotic exact finishing positions (which are easily ruined by a single bad pit stop or safety car), this model uses a multi-class classification approach to place drivers into one of three realistic finishing brackets:

*    **Podium:** P1 - P3
*    **Points:** P4 - P10
*    **Lower Bracket:** P11 - P20 (Includes DNFs)

### Input Features
The model makes its predictions based on 5 key pre-race metrics:
1.  **Qualifying Position:** The driver's raw qualifying result.
2.  **Grid Penalties:** Number of grid slots dropped (calculates the *effective* starting grid).
3.  **Quali Pace Delta:** The time gap (in seconds) between the driver's fastest lap and the pole-sitter's time.
4.  **Constructor Standing:** The team's overall rank (1st - 10th) going into the race weekend.
5.  **Team PPG:** The average Points Per Race scored by the constructor up to that point in the season.

## Model Architecture & Training

*   **Framework:** Built with TensorFlow/Keras.
*   **Architecture:** A lightweight feedforward Dense Neural Network (`32 -> 16` neurons with ReLU activation, Batch Normalization, and Dropout to prevent overfitting).
*   **Data Source:** Trained on official timing and standings data from the 2024 and 2025 F1 seasons using the [FastF1](https://docs.fastf1.dev/) library and Ergast API.
*   **Handling Racing Realities:** The dataset specifically accounts for "Survivor Bias" by mapping DNFs, crashes, and engine failures directly into the Lower Bracket, ensuring the model realistically evaluates midfield attrition.

## 💻 Tech Stack
*   **Frontend:** Streamlit
*   **Machine Learning:** TensorFlow, Scikit-Learn (StandardScaler)
*   **Data Processing:** Pandas, NumPy, FastF1
