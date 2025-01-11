# Heart Disease Prediction using ML and Flask

## File Structure
```
project/
│
├── app.py             # Flask application script
├── templates/
│   ├── index.html     # Prediction form page
│   ├── result.html    # Prediction result page
├── static/
│   └── style.css      # CSS for styling
├── models/
│   ├── voting_model.pkl  # Trained machine learning model
│   ├── scaler.pkl        # Feature scaler
├── requirements.txt   # Dependencies list
├── heart.csv          # Dataset file
└── README.md          # Project documentation
```

## Instructions
1. Clone repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run `app.py` using `python app.py`.
4. Access the app at `http://localhost:5000/`.
