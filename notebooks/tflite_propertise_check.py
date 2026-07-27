import tensorflow as tf
import numpy as np

# Load your TFLite model
interpreter = tf.lite.Interpreter(model_path="best.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("--- Input Details ---")
print(f"Name: {input_details[0]['name']}")
print(f"Shape: {input_details[0]['shape']}")  # Expected: [1, 640, 640, 3] or similar
print(f"Data Type: {input_details[0]['dtype']}")

print("\n--- Output Details ---")
for i, output in enumerate(output_details):
    print(f"Output {i}:")
    print(f"  Name: {output['name']}")
    print(f"  Shape: {output['shape']}")
    print(f"  Data Type: {output['dtype']}")