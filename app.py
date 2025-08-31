from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np
import pandas as pd
# NEW: Import the metrics we need for evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

app = Flask(__name__)

# Load your trained model and scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

# The list of 22 feature names your model was trained on
FEATURE_NAMES = [
    'SizeOfUninitializedData', 'SizeOfCode', 'NumberOfSections', 'Size', 
    'BaseOfData', 'DllCharacteristics', 'SizeOfImage', 'PE_TYPE', 
    'FileAlignment', 'Entropy', 'SizeOfInitializedData', 'PointerToSymbolTable', 
    'TimeDateStamp', 'Characteristics', 'SizeOfHeaders', 'BaseOfCode', 
    'Machine', 'NumberOfRvaAndSizes', 'ImageBase', 'NumberOfSymbols', 
    'Magic', 'SizeOfOptionalHeader'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)
    scaled_features = scaler.transform(final_features)
    prediction = model.predict(scaled_features)
    output = "Malware" if prediction[0] == 1 else "Goodware"
    return render_template('result.html', prediction_text=f'The file is predicted to be: {output}')

# MODIFIED: The upload function is now smarter
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)

    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)

            # --- NEW LOGIC STARTS HERE ---
            
            # Check if there is a 'Label' column for evaluation
            if 'Label' in df.columns:
                y_true = df['Label']
                X_features = df.drop('Label', axis=1)
                
                # Ensure feature columns are in the correct order
                if not all(col in X_features.columns for col in FEATURE_NAMES):
                    missing_cols = [col for col in FEATURE_NAMES if col not in X_features.columns]
                    return render_template('result.html', prediction_text=f"Error: Missing columns for evaluation: {', '.join(missing_cols)}")
                
                X_for_prediction = X_features[FEATURE_NAMES]
                
                # Scale features and make predictions
                scaled_features = scaler.transform(X_for_prediction)
                predictions = model.predict(scaled_features)
                
                # Calculate metrics
                accuracy = accuracy_score(y_true, predictions)
                auc = roc_auc_score(y_true, predictions)
                cm = confusion_matrix(y_true, predictions)
                
                return render_template('result.html', 
                                       evaluation=True,
                                       accuracy=f"{accuracy:.4f}", 
                                       auc=f"{auc:.4f}", 
                                       confusion_matrix=cm.tolist()) # Convert numpy array to list for HTML
            
            else:
                # --- OLD LOGIC (PREDICTION ONLY) ---
                if not all(col in df.columns for col in FEATURE_NAMES):
                    missing_cols = [col for col in FEATURE_NAMES if col not in df.columns]
                    return render_template('result.html', prediction_text=f"Error: Missing columns for prediction: {', '.join(missing_cols)}")

                df_for_prediction = df[FEATURE_NAMES]
                scaled_data = scaler.transform(df_for_prediction)
                predictions = model.predict(scaled_data)
                
                results = [f"Row {i+1}: {'Malware' if pred == 1 else 'Goodware'}" for i, pred in enumerate(predictions)]
                return render_template('result.html', results_list=results)

        except Exception as e:
            return render_template('result.html', prediction_text=f"An error occurred: {e}")

    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)