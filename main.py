import cv2
import pytesseract
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import threading
import pyttsx3
import numpy as np
import time
from tool.predictor import Predictor
import pygame
import traceback
import os

pytesseract.pytesseract.tesseract_cmd = r'F:\Tesseract-OCR\tesseract'
model_predictor = Predictor(device='cpu', model_type='seq2seq', weight_path='./weights/seq2seq_0.pth')
pyttsx3_engine = pyttsx3.init()
pyttsx3_engine.setProperty('voice', 'vietnam')
pyttsx3_engine.setProperty('rate', 145)

cap = cv2.VideoCapture('http://192.168.1.3:8080/video')
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
command=None
def text_correction(text):
    try:
        return model_predictor.predict(text.strip(), NGRAM=6)
    except Exception as e:
        print(f"Error: {str(e)}")
        return text

def preprocess_image(image):
    # Resize (optional)
    image = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("Resized.jpg", image)

    # # Apply noise removal
    # kernel = np.ones((3, 3), np.uint8)
    # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # cv2.imwrite("Noise removal.jpg", image)

    # # Contrast stretching
    # image = cv2.normalize(image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    # cv2.imwrite("Contrast stretching.jpg", image)

    # # Sharpening
    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    # image = cv2.filter2D(image, -1, kernel=kernel)
    # cv2.imwrite("Sharpening.jpg", image)

    # # Brightness adjustment
    # image = cv2.convertScaleAbs(image, alpha=1, beta=5)
    # cv2.imwrite("Brightness adjustment.jpg", image)

    # # Gamma correction
    # gamma = 1.5
    # lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # image = cv2.LUT(image, lookup_table)
    # cv2.imwrite("Gamma correction.jpg", image)

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("Gray.jpg", image)

    # # Apply binarization
    # _, image = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # cv2.imwrite("Binary.jpg", image)

    # # Contrast enhancement
    # image = cv2.equalizeHist(image)
    # cv2.imwrite("Contrast enhancement.jpg", image)

    # # Skew correction 
    # coords = np.column_stack(np.where(image > 0))
    # angle = cv2.minAreaRect(coords)[-1]
    # if angle < -45:
    #     angle = -(90 + angle)
    # else:
    #     angle = -angle
    # (h, w) = image.shape[:2]
    # center = (w // 2, h // 2)
    # M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # cv2.imwrite("Rotated.jpg", image)
    return image

def listen_for_command():
    global command
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
        print("Google is listening for command...")
    try:
        # command = r.recognize_whisper(audio, language='vietnamese')
        command = r.recognize_google(audio, language='vi-VN')
        # command=r.recognize_vosk(audio, language='vi-VN')
        print(f"Command received: {command}")
        return command
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return None

def speak_gtts(text):
    tts = gTTS(text=text, lang='vi')
    tts.save("temp.mp3")
    playsound(os.path.dirname(__file__) + "\\temp.mp3")
    print("Bot : Đã đọc xong")

def speak_pyttsx3(text):
    pyttsx3_engine.say(text)
    pyttsx3_engine.runAndWait()

thread=threading.Thread(target=listen_for_command)
thread.start()

while True:
    ret, frame = cap.read()
    cv2.imshow("Camera", frame)
    t=time.time()
    if command is not None and "đọc chữ" in command.lower():
        print("Đang tiền xử lý ảnh")
        frame=preprocess_image(frame)
        print("Bắt đầu đọc văn bản")
        recognized_text = pytesseract.image_to_string(frame, lang='vie')
        d = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT)
        text=d['text']
        n_boxes = len(d['text'])
        if n_boxes>0:
            print(recognized_text)
            for i in range(n_boxes):
                if int(d['conf'][i]) > 0:
                    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            frame = cv2.putText(frame, recognized_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite("frame with text.jpg", frame)
            cv2.imshow("Camera", frame)
            if recognized_text.strip():
                print("Văn bản chưa sửa lỗi: ", recognized_text)
                try:
                    recognized_text = text_correction(recognized_text)
                    print("Văn bản đã sửa lỗi: ", recognized_text)
                except Exception as e:
                    print(f"Error: {str(e)}")
                    traceback.print_exc()
                speak_thread = threading.Thread(target=speak_gtts, args=(recognized_text,))
                speak_thread.start()
                speak_thread.join()
        command = None
    else:
        if not thread.is_alive():
            thread=threading.Thread(target=listen_for_command)
            thread.start()
    if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:  
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()

