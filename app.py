import tensorflow as tf
import numpy as np
import cv2
import asyncio
import sounddevice as sd
from transformers import pipeline
from tensorflow.keras.models import load_model

# Загрузка модели насилия
print("Loading violence detection model...")
model = load_model('modelnew.h5')
model.save('model_saved_format', save_format='tf')
violence_model = tf.keras.models.load_model('model_saved_format')


ser_pipe = pipeline(model="Aniemore/wavlm-emotion-russian-resd")


async def run_violence_detection():
    video_stream = cv2.VideoCapture(0)  
    if not video_stream.isOpened():
        print("Error: Unable to access camera.")
        return

    while True:
        ret, frame = video_stream.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        
        resized_frame = cv2.resize(frame, (128, 128))   
        normalized_frame = resized_frame / 255.0  
        input_data = np.expand_dims(normalized_frame, axis=0)  

        try:
            predictions = violence_model.predict(input_data)
            violence_detected = predictions[0][0] > 0.5  

            
            if violence_detected:
                label = "Violence Detected"
                color = (0, 0, 255)  
            else:
                label = "No Violence Detected"
                color = (0, 255, 0)  

            
            cv2.putText(
                frame, label, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA
            )
        except Exception as e:
            print(f"Error in violence detection: {e}")

        cv2.imshow("Video Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        await asyncio.sleep(0)  
    video_stream.release()
    cv2.destroyAllWindows()  


# Функция для детекции эмоций
async def run_emotion_detection():
    sample_rate = 16000
    duration = 5

    while True:
        print("Recording audio...")
        
        
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        audio_data = audio_data.squeeze()  

        
        try:
            emotion = ser_pipe(audio_data, sampling_rate=sample_rate)[0]  
            print(f"Detected Emotion: {emotion['label']} with score {emotion['score']}")
        except Exception as e:
            print(f"Error in emotion detection: {e}")

        await asyncio.sleep(3)  


async def main():
    violence_task = asyncio.create_task(run_violence_detection())
    emotion_task = asyncio.create_task(run_emotion_detection())
    await asyncio.gather(violence_task, emotion_task)


if __name__ == "__main__":
    asyncio.run(main())
