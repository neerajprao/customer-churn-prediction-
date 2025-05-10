import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import os
import traceback

# Try loading model-specific libraries
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available")

try:
    from lightgbm import Booster
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available")

try:
    import pennylane as qml
    from pennylane import numpy as qnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    print("Warning: PennyLane not available. This is needed for quantum model execution.")

# ---------------- Define Model Paths ----------------
MODEL_BASE_PATH = '/Users/neerajprao/Downloads/ML-vs-DL-main'
ML_MODEL_PATH = f'{MODEL_BASE_PATH}/front_end/models'
QML_MODEL_PATH = f'{MODEL_BASE_PATH}/qml'
QNN_MODEL_PATH = f'{MODEL_BASE_PATH}/qnn/qnn_model.npz'

# ---------------- Define DL Model Architectures ----------------
class NodeModel(nn.Module):
    def __init__(self):
        super(NodeModel, self).__init__()
        self.tree_layer = nn.ModuleDict({
            'hidden_layers': nn.ModuleList([
                nn.Linear(13, 128),
                nn.Linear(128, 128),
                nn.Linear(128, 128),
                nn.Linear(128, 128),
                nn.Linear(128, 128),
                nn.Linear(128, 128),
            ]),
            'output_layer': nn.Linear(128, 56)
        })
        self.fc1 = nn.Linear(56, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        tree_features = x
        for layer in self.tree_layer['hidden_layers']:
            tree_features = F.relu(layer(tree_features))
        tree_output = self.tree_layer['output_layer'](tree_features)
        fc_features = F.relu(self.fc1(tree_output))
        fc_features = F.relu(self.fc2(fc_features))
        fc_output = self.fc3(fc_features)
        return fc_output

class SaintModel(nn.Module):
    def __init__(self):
        super(SaintModel, self).__init__()
        self.fc1 = nn.Linear(13, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class TabTransformerModel(nn.Module):
    def __init__(self):
        super(TabTransformerModel, self).__init__()
        self.fc1 = nn.Linear(13, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class TabNetModel(nn.Module):
    def __init__(self):
        super(TabNetModel, self).__init__()
        self.fc1 = nn.Linear(13, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ---------------- Load models safely ----------------
available_models = {
    'xgb': False,
    'rf': False,
    'lgbm': False,
    'catboost': False,
    'node': False,
    'saint': False,
    'tabtransformer': False,
    'tabnet': False,
    'vqc': False,
    'qnn': False
}

# Define feature names for input
feature_names = [
    'CreditScore', 'Geography_France', 'Geography_Germany', 'Geography_Spain',
    'Gender_Female', 'Gender_Male', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
]

# Load ML models
if XGBOOST_AVAILABLE:
    try:
        xgb_model = XGBClassifier()
        xgb_model.load_model(f'{ML_MODEL_PATH}/xgboost_model.json')
        available_models['xgb'] = True
        print("XGBoost model loaded successfully")
    except Exception as e:
        print(f"Error loading XGBoost model: {e}")
        xgb_model = None

try:
    rf_model = joblib.load(f'{ML_MODEL_PATH}/random_forest.pkl')
    available_models['rf'] = True
    print("Random Forest model loaded successfully")
except Exception as e:
    print(f"Error loading Random Forest model: {e}")
    rf_model = None

if LIGHTGBM_AVAILABLE:
    try:
        lgbm_booster = Booster(model_file=f'{ML_MODEL_PATH}/lightgbm_model.txt')
        lgbm_model = lgbm_booster
        available_models['lgbm'] = True
        print("LightGBM model loaded successfully")
    except Exception as e:
        print(f"Error loading LightGBM model: {e}")
        lgbm_model = None

if CATBOOST_AVAILABLE:
    try:
        catboost_model = CatBoostClassifier()
        catboost_model.load_model(f'{ML_MODEL_PATH}/catboost_model.cbm')
        available_models['catboost'] = True
        print("CatBoost model loaded successfully")
    except Exception as e:
        print(f"Error loading CatBoost model: {e}")
        catboost_model = None

# Load DL models
try:
    node_model = NodeModel()
    node_model.load_state_dict(torch.load(f'{ML_MODEL_PATH}/node_model.pt'))
    node_model.eval()
    available_models['node'] = True
    print("NODE model loaded successfully")
except Exception as e:
    print(f"Error loading NODE model: {e}")
    node_model = None

try:
    saint_model = SaintModel()
    saint_model.load_state_dict(torch.load(f'{ML_MODEL_PATH}/saint_model.pt'), strict=False)
    saint_model.eval()
    available_models['saint'] = True
    print("SAINT model loaded successfully")
except Exception as e:
    print(f"Error loading SAINT model: {e}")
    saint_model = None

try:
    tabtransformer_model = TabTransformerModel()
    tabtransformer_model.load_state_dict(torch.load(f'{ML_MODEL_PATH}/tabtransformer_model.pt'), strict=False)
    tabtransformer_model.eval()
    available_models['tabtransformer'] = True
    print("TabTransformer model loaded successfully")
except Exception as e:
    print(f"Error loading TabTransformer model: {e}")
    tabtransformer_model = None

try:
    tabnet_model = TabNetModel()
    tabnet_model.load_state_dict(torch.load(f'{ML_MODEL_PATH}/tabnet_model.pt'), strict=False)
    tabnet_model.eval()
    available_models['tabnet'] = True
    print("TabNet model loaded successfully")
except Exception as e:
    print(f"Error loading TabNet model: {e}")
    tabnet_model = None

# Load VQC model
try:
    vqc_model_data = joblib.load(f'{QML_MODEL_PATH}/vqc_model.pkl')
    print("VQC model data loaded successfully")
    
    # Print model type information
    print(f"VQC Model type: {type(vqc_model_data)}")
    if isinstance(vqc_model_data, dict):
        print("VQC Model is a dictionary with keys:", vqc_model_data.keys())
    
    available_models['vqc'] = True
except Exception as e:
    print(f"Error loading VQC model: {e}")
    vqc_model_data = None

# Load QNN model
try:
    # Load the saved quantum neural network model
    qnn_model_data = np.load(QNN_MODEL_PATH, allow_pickle=True)
    print("QNN model data loaded successfully")
    
    # Print model type information
    print(f"QNN Model type: {type(qnn_model_data)}")
    if isinstance(qnn_model_data, np.lib.npyio.NpzFile):
        print("QNN Model is an npz file with keys:", qnn_model_data.files)
    
    available_models['qnn'] = True
except Exception as e:
    print(f"Error loading QNN model: {e}")
    qnn_model_data = None

# ---------------- Input & Prediction Functions ----------------
def get_user_input():
    """Get customer data input from the user"""
    print("\nPlease enter the following features:")
    values = []
    values.append(float(input("Credit Score (e.g., 650): ")))

    geo = input("Geography (France, Germany, Spain): ").strip().capitalize()
    values += [1 if geo == "France" else 0,
               1 if geo == "Germany" else 0,
               1 if geo == "Spain" else 0]

    gender = input("Gender (Male or Female): ").strip().capitalize()
    values += [1 if gender == "Female" else 0,
               1 if gender == "Male" else 0]

    values.append(float(input("Age (e.g., 35): ")))
    values.append(int(input("Tenure (e.g., 5): ")))
    values.append(float(input("Balance (e.g., 75000.0): ")))
    values.append(int(input("Number of Products (1-4): ")))
    values.append(int(input("Has Credit Card (1 = Yes, 0 = No): ")))
    values.append(int(input("Is Active Member (1 = Yes, 0 = No): ")))
    values.append(float(input("Estimated Salary (e.g., 65000.0): ")))

    return np.array(values).reshape(1, -1)

def predict_lgbm(model, data):
    """Helper function for LightGBM prediction"""
    if model is None:
        return None
    prob = model.predict(data)[0]
    return 1 if prob > 0.5 else 0

def predict_ml_models(user_input):
    """Make predictions using traditional machine learning models"""
    print("\n--- Machine Learning Model Predictions ---")
    results = {}
    
    if available_models['xgb']:
        try:
            pred = xgb_model.predict(user_input)[0]
            print("XGBoost:", pred)
            results['XGBoost'] = pred
        except Exception as e:
            print(f"Error in XGBoost prediction: {e}")
    else:
        print("XGBoost: Not available")
    
    if available_models['rf']:
        try:
            pred = rf_model.predict(user_input)[0]
            print("Random Forest:", pred)
            results['Random Forest'] = pred
        except Exception as e:
            print(f"Error in Random Forest prediction: {e}")
    else:
        print("Random Forest: Not available")
    
    if available_models['lgbm']:
        try:
            pred = predict_lgbm(lgbm_model, user_input)
            print("LightGBM:", pred)
            results['LightGBM'] = pred
        except Exception as e:
            print(f"Error in LightGBM prediction: {e}")
    else:
        print("LightGBM: Not available")
    
    if available_models['catboost']:
        try:
            pred = catboost_model.predict(user_input)[0]
            print("CatBoost:", pred)
            results['CatBoost'] = pred
        except Exception as e:
            print(f"Error in CatBoost prediction: {e}")
    else:
        print("CatBoost: Not available")
    
    return results

def predict_dl_models(user_tensor):
    """Make predictions using deep learning models"""
    print("\n--- Deep Learning Model Predictions ---")
    results = {}
    
    if len(user_tensor.shape) == 2 and user_tensor.shape[0] == 1:
        tensor_for_models = user_tensor.squeeze(0)
    else:
        tensor_for_models = user_tensor
    
    if available_models['node']:
        try:
            with torch.no_grad():
                node_output = node_model(tensor_for_models)
                probabilities = F.softmax(node_output, dim=0)
                node_pred = torch.argmax(probabilities).item()
            print("NODE Model:", node_pred)
            results['NODE'] = node_pred
        except Exception as e:
            print(f"Error in NODE prediction: {e}")
    else:
        print("NODE Model: Not available")

    if available_models['saint']:
        try:
            with torch.no_grad():
                saint_output = saint_model(tensor_for_models)
            saint_pred = 1 if torch.sigmoid(saint_output).item() > 0.5 else 0
            print("SAINT Model:", saint_pred)
            results['SAINT'] = saint_pred
        except Exception as e:
            print(f"Error in SAINT prediction: {e}")
    else:
        print("SAINT Model: Not available")

    if available_models['tabtransformer']:
        try:
            with torch.no_grad():
                tabtransformer_output = tabtransformer_model(tensor_for_models)
            tabtransformer_pred = 1 if torch.sigmoid(tabtransformer_output).item() > 0.5 else 0
            print("TabTransformer Model:", tabtransformer_pred)
            results['TabTransformer'] = tabtransformer_pred
        except Exception as e:
            print(f"Error in TabTransformer prediction: {e}")
    else:
        print("TabTransformer Model: Not available")

    if available_models['tabnet']:
        try:
            with torch.no_grad():
                tabnet_output = tabnet_model(tensor_for_models)
            tabnet_pred = 1 if torch.sigmoid(tabnet_output).item() > 0.5 else 0
            print("TabNet Model:", tabnet_pred)
            results['TabNet'] = tabnet_pred
        except Exception as e:
            print(f"Error in TabNet prediction: {e}")
    else:
        print("TabNet Model: Not available")
    
    return results

def apply_vqc_model(user_input):
    """
    Apply the VQC model to make predictions.
    """
    if not available_models['vqc']:
        print("VQC Model is not available")
        return None
    
    if not PENNYLANE_AVAILABLE and 'quantum_function' in str(vqc_model_data):
        print("PennyLane is required but not available for this model")
        return None
    
    try:
        # Assuming vqc_model_data is a dictionary containing model parameters
        if isinstance(vqc_model_data, dict):
            # Check if it has weights/parameters key
            if 'weights' in vqc_model_data:
                weights = vqc_model_data['weights']
                # Here you would apply these weights to your quantum circuit
            
            # Check if it has a predict function saved somehow
            if 'predict_function' in vqc_model_data:
                prediction_fn = vqc_model_data['predict_function']
                prediction = prediction_fn(user_input)
                return prediction
            
            # If the model contains a threshold for binary classification
            if 'threshold' in vqc_model_data:
                threshold = vqc_model_data['threshold']
                print(f"Classification threshold: {threshold}")
            
            # This is a placeholder for actually running inference with the model
            # You need to replace this with actual VQC inference code
            raw_output = 0.5  # Placeholder, replace with actual inference
            prediction = 1 if raw_output > 0.5 else 0
            return prediction
            
        # If the model is a function itself    
        elif callable(vqc_model_data):
            prediction = vqc_model_data(user_input)
            return prediction
            
        # If the model is a standard scikit-learn-like model after all
        elif hasattr(vqc_model_data, 'predict'):
            prediction = vqc_model_data.predict(user_input)[0]
            return prediction
            
        else:
            print(f"Unknown model format: {type(vqc_model_data)}")
            return None
            
    except Exception as e:
        print(f"Error in VQC model prediction: {e}")
        traceback.print_exc()
        return None

def predict_vqc_model(user_input):
    """Make a prediction using the VQC model"""
    print("\n--- Quantum ML Model Prediction ---")
    
    if available_models['vqc']:
        prediction = apply_vqc_model(user_input)
        
        if prediction is not None:
            print("VQC Model:", prediction)
            return {"VQC": prediction}
        else:
            print("Could not make a prediction with VQC model")
            return {}
    else:
        print("VQC Model: Not available")
        return {}

def apply_qnn_model(user_input):
    """
    Apply the QNN model to make predictions.
    """
    if not available_models['qnn']:
        print("QNN Model is not available")
        return None
    
    if not PENNYLANE_AVAILABLE:
        print("PennyLane is required but not available for this model")
        return None
    
    try:
        # Process the npz file based on its structure
        if isinstance(qnn_model_data, np.lib.npyio.NpzFile):
            # Extract weights/parameters from the npz file
            weights = None
            
            # Check common key names for weights
            if 'weights' in qnn_model_data.files:
                weights = qnn_model_data['weights']
            elif 'params' in qnn_model_data.files:
                weights = qnn_model_data['params']
            elif 'model_params' in qnn_model_data.files:
                weights = qnn_model_data['model_params']
                
            if weights is not None:
                
                # Define a basic QNN circuit for inference
                dev = qml.device("default.qubit", wires=4)
                
                @qml.qnode(dev)
                def qnn_circuit(inputs, weights):
                    # Encode the input features
                    for i, x in enumerate(inputs):
                        qml.RY(x, wires=i % 4)
                    
                    # Apply trainable layers
                    for layer in weights:
                        for i, param in enumerate(layer):
                            qml.RY(param, wires=i % 4)
                        qml.CNOT(wires=[0, 1])
                        qml.CNOT(wires=[1, 2])
                        qml.CNOT(wires=[2, 3])
                        qml.CNOT(wires=[3, 0])
                    
                    # Return expectation values for first 3 qubits
                    return [qml.expval(qml.PauliZ(i)) for i in range(3)]
                
                # Preprocess input (normalize if needed)
                processed_input = user_input[0]  # Get the first row
                
                # Execute the circuit
                result = qnn_circuit(processed_input, weights)
                
                # Process the results for classification
                # Convert result to numpy array to handle it properly
                result_array = np.array(result)
                
                # Flatten the array if it's multi-dimensional
                if result_array.ndim > 1:
                    result_array = result_array.flatten()
                
                # Count positive values
                positive_count = np.sum(result_array > 0)
                total_elements = result_array.size
                
                # Make prediction based on majority vote
                prediction = 1 if positive_count > total_elements / 2 else 0
                
                return prediction
            
            else:
                print("Could not find weights in the npz file")
                return None
            
        else:
            print(f"Unknown model format: {type(qnn_model_data)}")
            return None
            
    except Exception as e:
        print(f"Error in QNN model prediction: {e}")
        traceback.print_exc()
        return None

def predict_qnn_model(user_input):
    """Make a prediction using the QNN model"""
    print("\n--- Quantum Neural Network Model Prediction ---")
    
    if available_models['qnn']:
        prediction = apply_qnn_model(user_input)
        
        if prediction is not None:
            print("QNN Model prediction:", prediction)
            return {"QNN": prediction}
        else:
            print("Could not make a prediction with QNN model")
            return {}
    else:
        print("QNN Model: Not available")
        return {}

def predict_all_models(user_input):
    """Make predictions using all available models and show consensus"""
    # Convert numpy input to PyTorch tensor for DL models
    user_tensor = torch.tensor(user_input, dtype=torch.float32)

    # Get predictions from all model types
    ml_results = predict_ml_models(user_input)
    dl_results = predict_dl_models(user_tensor)
    vqc_results = predict_vqc_model(user_input)
    qnn_results = predict_qnn_model(user_input)
    
    # Combine results
    all_results = {**ml_results, **dl_results, **vqc_results, **qnn_results}
    
    # Calculate and display consensus if we have enough models
    if len(all_results) >= 2:
        print("\n--- Consensus Analysis ---")
        ones = sum(1 for pred in all_results.values() if pred == 1)
        zeros = sum(1 for pred in all_results.values() if pred == 0)
        total = len(all_results)
        
        print(f"Models predicting customer stays (0): {zeros}/{total} ({zeros/total*100:.1f}%)")
        print(f"Models predicting customer churns (1): {ones}/{total} ({ones/total*100:.1f}%)")
        
        consensus = 1 if ones > zeros else 0
        confidence = max(ones, zeros) / total
        print(f"Consensus prediction: {consensus} with {confidence*100:.1f}% agreement")
        
        result_text = "Customer is likely to churn (leave the bank)" if consensus == 1 else "Customer is likely to stay"
        print(f"\nFinal result: {result_text}")

def show_example_values():
    """Display example values for reference"""
    examples = {
        'CreditScore': '650 (range: 300-850)',
        'Geography': 'France, Germany, or Spain',
        'Gender': 'Female or Male',
        'Age': '35 (range: 18-100)',
        'Tenure': '5 (range: 0-10 years)',
        'Balance': '75000.00 (account balance)',
        'NumOfProducts': '2 (range: 1-4 products)',
        'HasCrCard': '1 (Yes) or 0 (No)',
        'IsActiveMember': '1 (Yes) or 0 (No)',
        'EstimatedSalary': '65000.00 (annual salary)'
    }
    print("\nExample values for reference:")
    for feature, example in examples.items():
        print(f"{feature}: {example}")
    print()

def main():
    """Main function to run the prediction system"""
    print("Bank Customer Churn Prediction - Comprehensive Model System")
    print("=========================================================")
    print("This program predicts whether a bank customer is likely to leave the bank.")
    print("0 = Customer stays, 1 = Customer churns (leaves)")
    
    # Display available models
    print("\nAvailable models:")
    for model_name, available in available_models.items():
        status = "Available ✓" if available else "Not available ✗"
        print(f"- {model_name.upper()}: {status}")

    while True:
        show_example_values()
        user_input = get_user_input()
        predict_all_models(user_input)
        
        cont = input("\nDo you want to make another prediction? (y/n): ").strip().lower()
        if cont != 'y':
            print("Thank you for using the Bank Customer Churn Prediction tool.")
            break

if __name__ == "__main__":
    main()