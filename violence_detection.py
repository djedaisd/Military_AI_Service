import cv2
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your pre-trained model
print("Loading model ...")
model = load_model('Violence Detection\\modelnew.h5')
model.save('model_saved_format', save_format='tf')
model = tf.keras.models.load_model('model_saved_format')
Q = deque(maxlen=128)  # Queue to store predictions for averaging

# Commented out webcam capture for future use
# vs = cv2.VideoCapture(0)
video_path = r'C:\Users\nrs12\OneDrive\Рабочий стол\violence\Violence-Alert-System\Violence Detection\Testing videos\IMG_8132.MP4'  # Replace this with your video file path
vs = cv2.VideoCapture(video_path)

(W, H) = (None, None)
trueCount = 0

while True:
    # Read frame from the video file (or webcam when uncommented)
    (grabbed, frame) = vs.read()
    
    # If frame is not grabbed, break the loop
    if not grabbed:
        break
    
    # If frame dimensions are empty, initialize them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # Clone the frame for output display, and preprocess it for prediction
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (128, 128)).astype("float32")
    frame = frame.reshape(128, 128, 3) / 255

    # Make predictions on the frame
    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    Q.append(preds)  # Add predictions to the queue

    # Perform prediction averaging over the current history of previous predictions
    results = np.array(Q).mean(axis=0)
    label = int((results > 0.5)[0])  # 1 for violence, 0 for non-violence

    # Set color based on prediction
    text_color = (0, 0, 255) if label == 1 else (0, 255, 0)
    if label == 1:
        trueCount += 1

    # Display the label on the output frame
    text = "Violence Detected" if label else "No Violence"
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(output, text, (35, 50), FONT, 1.25, text_color, 3)

    # Show the output frame in real-time
    cv2.imshow("Real-Time Violence Detection", output)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close display windows
vs.release()
cv2.destroyAllWindows()
