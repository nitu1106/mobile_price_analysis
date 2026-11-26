"""
Task 2: Compare accuracy and inference speed for simple model compression ideas.
- Smaller Random Forest: fewer/smaller trees (tree-size reduction, pruning-like)
- Quantization: ONNX Runtime dynamic quantization of a trained Random Forest
- Knowledge distillation: small Decision Tree learns the Random Forest's predictions
"""
import os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

OUTPUT_DIR = "task74_compression_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv("train.csv")
if "price_range" not in df.columns:
    raise ValueError("train.csv must contain 'price_range'.")
X = df.drop(columns=["price_range", "id"], errors="ignore")
y = df["price_range"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def latency_ms(model, data, repeats=10):
    start = time.perf_counter()
    for _ in range(repeats):
        model.predict(data)
    return (time.perf_counter() - start) * 1000 / (repeats * len(data))


baseline = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)


small_forest = RandomForestClassifier(
    n_estimators=30, max_depth=8, random_state=42, n_jobs=-1
)
small_forest.fit(X_train, y_train)

# Knowledge distillation: train a small tree to copy the teacher's predictions.
teacher_train_predictions = baseline.predict(X_train)
student = DecisionTreeClassifier(max_depth=5, random_state=42)
student.fit(X_train, teacher_train_predictions)

rows = []
def add_result(name, model, prediction, note):
    rows.append({
        "Method": name,
        "Test_Accuracy": accuracy_score(y_test, prediction),
        "Agreement_with_baseline": accuracy_score(baseline_pred, prediction),
        "Latency_ms_per_row": latency_ms(model, X_test),
        "Model_size_KB": len(__import__("pickle").dumps(model)) / 1024,
        "Note": note
    })

add_result("Baseline Random Forest", baseline, baseline_pred, "Reference model")
add_result("Smaller Random Forest", small_forest, small_forest.predict(X_test),
           "Fewer trees and limited depth; pruning-like size reduction")
add_result("Distilled Decision Tree", student, student.predict(X_test),
           "Small student trained on teacher predictions")

# Quantize the baseline model using ONNX Runtime, if supported by installed versions.
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType

    onnx_path = os.path.join(OUTPUT_DIR, "baseline_forest.onnx")
    quant_path = os.path.join(OUTPUT_DIR, "baseline_forest_quantized.onnx")
    onnx_model = convert_sklearn(
        baseline, initial_types=[("input", FloatTensorType([None, X_train.shape[1]]))]
    )
    onnx.save_model(onnx_model, onnx_path)
    quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QInt8)
    import onnxruntime as ort
    session = ort.InferenceSession(quant_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    quant_pred = session.run(None, {input_name: X_test.to_numpy().astype(np.float32)})[0]
    class_labels = np.asarray(baseline.classes_)
    # ONNX classifier may return labels directly or probabilities; normalize common output.
    if quant_pred.ndim == 2:
        quant_pred = class_labels[np.argmax(quant_pred, axis=1)]
    else:
        quant_pred = quant_pred.reshape(-1)
    start = time.perf_counter()
    for _ in range(10):
        session.run(None, {input_name: X_test.to_numpy().astype(np.float32)})
    quant_latency = (time.perf_counter() - start) * 1000 / (10 * len(X_test))
    rows.append({
        "Method": "ONNX dynamic quantization",
        "Test_Accuracy": accuracy_score(y_test, quant_pred),
        "Agreement_with_baseline": accuracy_score(baseline_pred, quant_pred),
        "Latency_ms_per_row": quant_latency,
        "Model_size_KB": os.path.getsize(quant_path) / 1024,
        "Note": "Quantized ONNX model; quantization support can depend on package versions"
    })
except Exception as error:
    print("ONNX quantization skipped:", error)
    print("Install the ONNX packages listed in requirements.txt and try again.")

results = pd.DataFrame(rows)
results.to_csv(os.path.join(OUTPUT_DIR, "compression_comparison.csv"), index=False)
print(results.to_string(index=False))
print("\nSaved results in:", OUTPUT_DIR)
print("Note: sklearn Random Forest does not support direct neural-network pruning or distillation.")
