import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the model
with open("models/slang_classifier.pkl", "rb") as f:
    model = pickle.load(f)

slang_to_standard = model["slang_to_standard"]

# Load your test data
with open("data/test_data.json", "r") as f:
    test_data = json.load(f)

# Extract inputs and true labels
inputs = [item["slang"] for item in test_data]
true_labels = [item["standard"] for item in test_data]

# Predict using the pipeline
predicted_labels = slang_to_standard.predict(inputs)

# Evaluate
print("\n📊 Model Evaluation Metrics:\n")
print(f"✅ Accuracy:  {accuracy_score(true_labels, predicted_labels):.2f}")
print(f"✅ Precision: {precision_score(true_labels, predicted_labels, average='macro'):.2f}")
print(f"✅ Recall:    {recall_score(true_labels, predicted_labels, average='macro'):.2f}")
print(f"✅ F1 Score:  {f1_score(true_labels, predicted_labels, average='macro'):.2f}")
