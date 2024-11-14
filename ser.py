import time
import numpy as np
import sounddevice as sd
import librosa
from transformers import pipeline
import threading


pipe = pipeline("audio-classification", model="Aniemore/wavlm-emotion-russian-resd")

# Флаг для остановки потока
stop_thread = False


def record_audio(duration=5, samplerate=16000):
    
    print("Recording...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  
    return audio_data.flatten() 

# Функция для распознавания эмоций
def recognize_emotions():
    global stop_thread
    while not stop_thread:
        
        audio_data = record_audio(duration=5)
        
        
        predictions = pipe(audio_data)
        
        
        print(f"Predictions: {predictions}")
        
        time.sleep(5)  
        
# Запуск потока для распознавания
def start_recognition():
    # Используем поток для выполнения функции recognize_emotions параллельно
    recognition_thread = threading.Thread(target=recognize_emotions)
    recognition_thread.start()

# Функция для остановки записи
def stop_recognition():
    global stop_thread
    stop_thread = True
    print("Recognition stopped.")


if __name__ == "__main__":
    start_recognition()
    
   
    try:
        while True:
            user_input = input("Press 'q' to stop: ")
            if user_input.lower() == 'q':
                stop_recognition()
                break
    except KeyboardInterrupt:
        stop_recognition()
