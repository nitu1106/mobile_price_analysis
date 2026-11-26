"""
Task 1: Convert models to ONNX and TensorFlow Lite, then compare prediction
accuracy and inference latency.

"""
import time
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

CSV_FILE = "train.csv"
OUTPUT_DIR = "task73_conversion_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_FILE)
if "price_range" not in df.columns:
    raise ValueError("train.csv must contain the 'price_range' target column.")
X = df.drop(columns=["price_range", "id"], errors="ignore")
y = df["price_range"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train a small neural network. The same network is exported to both formats.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)
model = MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=42)
model.fit(X_train_scaled, y_train)
baseline_pred = model.predict(X_test_scaled)
baseline_acc = accuracy_score(y_test, baseline_pred)

# Export sklearn MLP to ONNX.
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

initial_type = [("float_input", FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)
onnx_path = os.path.join(OUTPUT_DIR, "mobile_price_model.onnx")
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
onnx_pred = session.run(None, {input_name: X_test_scaled})[0]
onnx_acc = accuracy_score(y_test, onnx_pred)

# Train equivalent Keras network and convert it to TensorFlow Lite.
import tensorflow as tf
keras_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(len(np.unique(y)), activation="softmax")
])
keras_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
keras_model.fit(X_train_scaled, y_train.to_numpy(), epochs=30, batch_size=32, verbose=0)
keras_prob = keras_model.predict(X_test_scaled, verbose=0)
keras_pred = np.argmax(keras_prob, axis=1)
keras_acc = accuracy_score(y_test, keras_pred)

converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_bytes = converter.convert()
tflite_path = os.path.join(OUTPUT_DIR, "mobile_price_model.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_bytes)

interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
input_info = interpreter.get_input_details()[0]
output_info = interpreter.get_output_details()[0]
tflite_pred = []
for row in X_test_scaled:
    interpreter.set_tensor(input_info["index"], row.reshape(1, -1).astype(input_info["dtype"]))
    interpreter.invoke()
    probabilities = interpreter.get_tensor(output_info["index"])[0]
    tflite_pred.append(int(np.argmax(probabilities)))
tflite_acc = accuracy_score(y_test, tflite_pred)

def measure_ms(predict_function, data, repeats=10):
    start = time.perf_counter()
    for _ in range(repeats):
        predict_function(data)
    return (time.perf_counter() - start) * 1000 / (repeats * len(data))

# Batch latency (ms per row); TFLite is measured row-by-row.
onnx_latency = measure_ms(lambda a: session.run(None, {input_name: a})[0], X_test_scaled)
sklearn_latency = measure_ms(model.predict, X_test_scaled)
start = time.perf_counter()
for row in X_test_scaled:
    interpreter.set_tensor(input_info["index"], row.reshape(1, -1).astype(input_info["dtype"]))
    interpreter.invoke()
tflite_latency = (time.perf_counter() - start) * 1000 / len(X_test_scaled)

results = pd.DataFrame([
    {"Format": "Original scikit-learn", "Accuracy": baseline_acc, "Latency_ms_per_row": sklearn_latency},
    {"Format": "ONNX Runtime (CPU)", "Accuracy": onnx_acc, "Latency_ms_per_row": onnx_latency},
    {"Format": "Keras reference", "Accuracy": keras_acc, "Latency_ms_per_row": np.nan},
    {"Format": "TensorFlow Lite (CPU)", "Accuracy": tflite_acc, "Latency_ms_per_row": tflite_latency},
])
results.to_csv(os.path.join(OUTPUT_DIR, "format_comparison.csv"), index=False)
print(results.to_string(index=False))
print("\nSaved ONNX and TFLite models in:", OUTPUT_DIR)
print("Note: this run measures CPU only. Run separately on other hardware to compare platforms.")


# Remember: ONNX -> open neural network exchange  is a model format; ONNX Runtime is the tool that runs ONNX models. TFLite is another format designed for lightweight deployment, especially on mobile and edge devices