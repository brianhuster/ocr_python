import cv2
from PIL import Image
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import threading
import numpy as np
import time
from tool.predictor import Predictor as Corrector
import pytesseract
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import traceback
import os
import argparse
import urllib.request

pytesseract.pytesseract.tesseract_cmd = r'F:\Tesseract-OCR\tesseract' #replace it with the actual path to your tesseract file. If you are using MacOS or Linux, remove this line
text_corrector = Corrector(device='cpu', model_type='seq2seq', weight_path='./weights/seq2seq_0.pth')
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'https://huggingface.co/brianhuster/VietnameseOCR/resolve/main/vgg_transformer.pth'
config['device'] = 'cpu'
config['cnn']['pretrained']=False
config['predictor']['beamsearch']=False
OCRpredictor = Predictor(config)

def listen_for_command():
    global command
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("\nGoogle is listening for command...\n")
        audio = r.listen(source)
    try:
        command = r.recognize_google(audio, language='vi-VN')
        print(f"\nCommand received: {command}\n")
        return command
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return None

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

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binarization
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.waitKey(0)

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

def OCR(image):
    t=time.time()
    d = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    print("\nDetecting time: ", time.time()-t)
    t=time.time()
    n_boxes = len(d['text'])
    if n_boxes == 0:
        print("\nNo text detected\n")
        return
    text = ""
    last_y = d['top'][0]
    last_h = d['height'][0]
    for i in range(n_boxes):
        if int(d['conf'][i]) > 0:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            cut = image[y:y+h, x:x+w]
            cut = preprocess_image(cut)
            cv2.waitKey(0)
            cut_text = OCRpredictor.predict(img = Image.fromarray(cut))
            if y - last_y > last_h:
                text += "\n"
            last_y = y
            last_h = h
            text += cut_text + " "
    print("\nRecognization time: ", time.time()-t)
    return text

def text_correction(text):
    try:
        print("\nRecognized text: ", text)
        t=time.time()
        text=text_corrector.predict(text.strip(), NGRAM=6)
        print("\nCorrected text: ", text)
        print("\nCorrection time: ", time.time()-t)
        return text
    except Exception as e:
        print(f"Error: {str(e)}")
        return text

def speak_gtts(text):
    tts = gTTS(text=text, lang='vi')
    tts.save("temp.mp3")
    playsound("temp.mp3")
    os.remove("temp.mp3")

parser = argparse.ArgumentParser(description='Read text from an image or camera. You can only provide either --ImagePath or --CameraSource, not both')
parser.add_argument('-i', '--ImagePath', type=str, help='Path to an image')
parser.add_argument('-c', '--CameraSource', type=str, help='Index or URL of the camera to use. Default is 0')
args = parser.parse_args()

if args.ImagePath is not None and args.CameraSource is not None:
    parser.error('You can only provide either --ImagePath or --CameraSource, not both')
if args.ImagePath is None and args.CameraSource is None:
    args.CameraSource = '0'

if args.ImagePath:
    if args.ImagePath.startswith('http://') or args.ImagePath.startswith('https://'):
        resp = urllib.request.urlopen(args.ImagePath)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(args.ImagePath)
    recognized_text = OCR(image)
    if recognized_text.strip():
        recognized_text = text_correction(recognized_text)
        speak_gtts(recognized_text)
    exit()

if args.CameraSource:
    if args.CameraSource.isdigit():
        cap = cv2.VideoCapture(int(args.CameraSource))
    else:
        cap = cv2.VideoCapture(args.CameraSource)
    cap.set(3, 640)
    cap.set(4, 560)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    command=None

    thread=threading.Thread(target=listen_for_command)
    thread.start()

    while True:
        ret, frame = cap.read()
        cv2.imshow("Camera", frame)
        t=time.time()
        if command is not None:
            if "đọc chữ" in command.lower():
                recognized_text = OCR(frame)
                cv2.imshow("Camera", frame)
                if recognized_text.strip():
                    recognized_text = text_correction(recognized_text)
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

