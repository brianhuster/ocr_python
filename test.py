from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import cv2
import pytesseract
from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2
from gtts import gTTS
from playsound import playsound

pytesseract.pytesseract.tesseract_cmd = r'F:\Tesseract-OCR\tesseract'

config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'https://huggingface.co/brianhuster/VietnameseOCR/resolve/main/vgg_transformer.pth'
# config['weights'] = 'F:/My projects/SCIC/vgg_transformer.pth'
config['device'] = 'cpu'
config['cnn']['pretrained']=False
config['predictor']['beamsearch']=False
detector = Predictor(config)

def preprocess_image(image):
    # Resize 
    image = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("Resized.jpg", image)

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("Gray.jpg", image)

    return image

def OCR(image):
    d = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    n_boxes = len(d['text'])
    if n_boxes == 0:
        print("No text detected")
        return
    text = ""
    last_y = d['top'][0]
    last_h = d['height'][0]
    for i in range(n_boxes):
        if int(d['conf'][i]) > 0:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            cut = image[y:y+h, x:x+w]
            cut = preprocess_image(cut)
            cv2.imshow("cut.jpg", cut)
            cv2.waitKey(0)
            cut_text = detector.predict(img = Image.fromarray(cut))
            if y - last_y > last_h:
                text += "\n"
            last_y = y
            last_h = h
            text += cut_text + " "
    return text

def speak_gtts(text):
    tts = gTTS(text=text, lang='vi')
    tts.save("temp.mp3")
    playsound("temp.mp3")

img = cv2.imread('trangsach.jpg')
# img = preprocess_image(img)
recognized_text = OCR(img)
print(recognized_text)
# speak_gtts(recognized_text)