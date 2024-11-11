import onnx
from onnx_tf.backend import prepare

# Load your Keras model
model = tf.keras.models.load_model("modelnew.h5")

# Convert to ONNX format
onnx_model = prepare(model).export_graph()

# Save the ONNX model
onnx.save_model(onnx_model, "modelnew.onnx")
