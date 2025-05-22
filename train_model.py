import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Set base project path
project_path = os.getcwd()  # Use current directory
data_path = os.path.join(project_path, "data", "census.csv")
print(f"Loading data from: {data_path}")

# Load the census data
data = pd.read_csv(data_path)

# Split the provided data into training and testing
train, test = train_test_split(data, test_size=0.20, random_state=42)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process the training data
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Process the test data
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# Train the model
model = train_model(X_train, y_train)

# Create model directory if it doesn't exist
model_dir = os.path.join(project_path, "model")
os.makedirs(model_dir, exist_ok=True)

# Save model and encoders
model_path = os.path.join(model_dir, "model.pkl")
encoder_path = os.path.join(model_dir, "encoder.pkl")
lb_path = os.path.join(model_dir, "label_binarizer.pkl")

save_model(model, model_path)
save_model(encoder, encoder_path)
save_model(lb, lb_path)

print(f"✅ Model saved to {model_path}")
print(f"✅ Encoder saved to {encoder_path}")
print(f"✅ Label binarizer saved to {lb_path}")

# Load model
model = load_model(model_path)

# Run inference
preds = inference(model, X_test)

# Calculate and print overall performance
p, r, fb = compute_model_metrics(y_test, preds)
print(f"\n📈 Overall Model Performance:")
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Evaluate performance on data slices
slice_output_path = os.path.join(project_path, "slice_output.txt")
with open(slice_output_path, "w") as f:
    for col in cat_features:
        for slice_value in sorted(test[col].unique()):
            count = test[test[col] == slice_value].shape[0]
            p, r, fb = performance_on_categorical_slice(
                data=test,
                column_name=col,
                slice_value=slice_value,
                categorical_features=cat_features,
                label="salary",
                encoder=encoder,
                lb=lb,
                model=model
            )
            if p is not None:
                f.write(f"{col}: {slice_value}, Count: {count:,}\n")
                f.write(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}\n\n")

print(f"\n📄 Slice metrics written to: {slice_output_path}")
