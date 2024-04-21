import cv2
import pytesseract
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import threading
import pyttsx3
import numpy as np
from tool.predictor import Predictor
import easyocr
import matplotlib.pyplot as plt
from PIL import Image
import pygame

from vietocr.tool.predictor import Predictor as OCR
from vietocr.tool.config import Cfg
pygame.mixer.init()
pygame.mixer.music.load('temp.mp3')
pygame.mixer.music.play()
pytesseract.pytesseract.tesseract_cmd = r'F:\Tesseract-OCR\tesseract'
def text_correction(text):
    model_predictor = Predictor(device='cpu', model_type='seq2seq', weight_path='./weights/seq2seq_0.pth')
    return model_predictor.predict(text.strip(), NGRAM=6)

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
    # print (angle)
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
image = cv2.imread("anime.jpg", cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = preprocess_image(image)
recognized_text = pytesseract.image_to_string(image, lang='vie')
print("Văn bản : " + recognized_text)
if recognized_text:
    recognized_text = text_correction(recognized_text)
print(recognized_text)
cv2.imshow("Image", image)