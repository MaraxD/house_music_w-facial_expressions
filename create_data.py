import tensorflow
from tensorflow import keras
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import os
import cv2
import numpy as np
import pygame
import time
import threading
from queue import Queue


threads=[]


pygame.mixer.init()

class SoundPlayer():
    def __init__(self):
        self.current_thread = None
        self.current_emotion = None
        self.lock = threading.Lock()

    def play_sound(self, emotion: str) -> None:
        with self.lock:
            self.current_emotion=emotion
            if self.current_thread and self.current_thread.is_alive():
                self.current_thread.do_run=False
                pygame.mixer.stop()
                self.current_thread.join()

            self.current_thread=threading.Thread(target=self.play_sound_thread, args=(emotion,))
            self.current_thread.start()

    def play_sound_thread(self, emotion: str):
        t=threading.current_thread()

        try:
            emotion_sound=pygame.mixer.Sound(f"./house_sounds/{emotion.lower()}.mp3")
            emotion_sound.play()

            sleep_time=0
            while sleep_time < emotion_sound.get_length() and getattr(t, "do_run", True):
                time.sleep(0.1)
                sleep_time+=0.1

        except Exception as e:
            print(f"could not play sound {e}")


face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier =load_model(r'model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)
sound_player=SoundPlayer()

while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            # play the sound with the name label.lower().mp3
            # the neutral sound will always play, also add a transition between the sounds
            # the sounds will always be looped until the emotion changes

            if label != sound_player.current_emotion:
                sound_player.play_sound(label)

            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if sound_player.current_thread and sound_player.current_thread.is_alive():
    sound_player.current_thread.do_run=False
    sound_player.current_thread.join()