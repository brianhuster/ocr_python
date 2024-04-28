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
import easyocr
import platform
import subprocess
import warnings
import pycountry
import pkg_resources
import sys

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

def run_shell_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    return output.decode(), error.decode()

def preprocess_image(image):
    # Resize
    image = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

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

def detect_tesseract(image): # detect using tesseract
    d = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    n_boxes = len(d['text'])
    if n_boxes == 0:
        print("\nNo text detected\n")
        return
    array=[]
    for i in range(n_boxes):
        if int(d['conf'][i]) > 0:
            array.append((d['top'][i], d['top'][i]+d['height'][i], d['left'][i], d['left'][i]+d['width'][i], d['text'][i]))
    return array

def detect_easyocr(image): # detect using easyocr
    t=time.time()
    d=reader.readtext(image)
    t=time.time()
    n_boxes = len(d)
    if n_boxes == 0:
        print("\nNo text detected\n")
        return
    array=[]
    for i in range(n_boxes):
        if d[i][2] > 0:
            (top, bottom, left, right, text) = (d[i][0][0][1], d[i][0][2][1], d[i][0][0][0], d[i][0][1][0], d[i][1])
            array.append((top, bottom, left, right, text))
    return array

def recognize(image, array):
    text=""
    last_bottom=array[0][1]
    for box in array:
        cut = image[int(box[0]):int(box[1]), int(box[2]):int(box[3])]
        cut = preprocess_image(cut)
        if args.recognizer == args.detector:
            cut_text=box[4]
        elif args.recognizer == 'easy_ocr':
            cut_text = '\n'.join(reader.readtext(cut, detail=0, paragraph=True))
        elif args.recognizer == 'vietocr':
            cut_text = vietocr_predictor.predict(img = Image.fromarray(cut))
        elif args.recognizer == 'tesseract':
            cut_text = pytesseract.image_to_string(cut, lang='vie')
        if box[0]>last_bottom:
            text += "\n"
        last_bottom = box[1]
        text += cut_text + " "
    return text

def text_correction(text):
    if not text:
        return text
    try:
        t=time.time()
        text=text_corrector.predict(text.strip(), NGRAM=6)
        return text
    except Exception as e:
        return text

def OCR(image):
    t=time.time()
    if args.detector == 'easyocr':
        array = detect_easyocr(image)
    elif args.detector == 'tesseract':
        array = detect_tesseract(image)
    if array is None:
        return
    text=recognize(image, array)
    if args.textCorrector == 'y' and args.language == 'vi' and text.strip():
        text=text_correction(text)
    print("Recognized text :", text)
    print(f"Time taken to process: {time.time()-t:.2f} seconds")
    return text

def speak_gtts(text):
    tts = gTTS(text=text, lang=args.language)
    tts.save("temp.mp3")
    playsound("temp.mp3")
    os.remove("temp.mp3")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read text from image(s) or camera. Among  --ImagePath (-i), imageFolders (-f) and --CameraSource (-c), you can provide ONLY ONE option for a runtime.')
    parser.add_argument('-i', '--ImagePath', type=str, help='Local path or URL to an image')
    parser.add_argument('-f', '--imagesFolder', type=str, help='Local path to a folder containing images. This will process all images in the folder and save recognized text in txt files the same folder.')
    parser.add_argument('-c', '--CameraSource', type=str, help='Index or URL of the video camera to use. Default is 0')
    if platform.system() != 'Windows':
        parser.add_argument('-d', '--detector', type=str, default='easyocr', help='Detector to use, "tesseract" or "easyocr". Default is "easyocr"')
        parser.add_argument('-r', '--recognizer', type=str, default='easyocr', help='Recognizer to use, "easy_ocr" or "vietocr". Default is "easy_ocr"')
    else:
        parser.add_argument('-d', '--detector', type=str, default='easyocr', help='Currently only "easyocr" is supported.')
        parser.add_argument('-r', '--recognizer', type=str, default='easyocr', help='Recognizer to use, "easy_ocr", "vietocr" or "tesseract". Default is "easy_ocr"')
    parser.add_argument('-l', '--language', type=str, default='vi', help='2-letter code of language to use for OCR. Default is "vi" (Vietnamese)')
    parser.add_argument('-t', '--textCorrector', type=str, default='n', help='Use text corrector or not (y/n). This only works for Vietnamese. Default is "n", however, we recommend using it if you are trying to recognize Vietnamese text.')
    parser.add_argument('-s', '--TTS', type=str, default='y', help='Use text-to-speech or not (y/n). Default is "y"')
    args = parser.parse_args()

    if args.recognizer == 'easyocr' or args.detector == 'easyocr':
        reader = easyocr.Reader([args.language,'en'])

    if args.recognizer == 'tesseract' or args.detector == 'tesseract':
        if platform.system() == 'Windows':
            parser.error("Sorry, Tesseract is not supported on Windows")
        print("\nCHECKING AND INSTALLING TESSERACT (if not found)\n")
        if platform.system() == 'Linux':
            run_shell_command("sudo apt-get install tesseract-ocr-vie")
            run_shell_command("sudo apt-get install tesseract-ocr-script-viet")
        elif platform.system() == 'Darwin':
            run_shell_command("sudo port install tesseract-vie")

    if args.recognizer == 'vietocr':
        if args.language != 'vi':
            parser.error("VietOCR only supports Vietnamese language.")
        config = Cfg.load_config_from_name('vgg_transformer')
        config['weights'] = 'https://huggingface.co/brianhuster/VietnameseOCR/resolve/main/vgg_transformer.pth'
        config['device'] = 'cpu'
        config['cnn']['pretrained']=False
        config['predictor']['beamsearch']=False
        vietocr_predictor = Predictor(config)

    if args.textCorrector == 'y':
        text_corrector = Corrector(device='cpu', model_type='seq2seq', weight_path=('weights/seq2seq_0.pth'))

    if args.language != 'vi' and args.textCorrector == 'y':
        warnings.warn("Text corrector only works for Vietnamese, so it will be disabled.")

    if (args.ImagePath is not None and args.CameraSource is not None) or (args.folderPath is not None and args.CameraSource is not None) or (args.folderPath is not None and args.ImagePath is not None):
        parser.error('You can provide ONLY ONE of these 3 options, either --ImagePath (-i), imageFolders (-f) or --CameraSource (-c) for a runtime.')
    if args.ImagePath is None and args.CameraSource is None:
        args.CameraSource = '0'

    #start processing

    if args.ImagePath:
        if args.ImagePath.startswith('http://') or args.ImagePath.startswith('https://'):
            try:
                resp = urllib.request.urlopen(args.ImagePath)
                image = np.asarray(bytearray(resp.read()), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            except urllib.error.HTTPError as e:
                if e.code == 403:
                    sys.exit("HTTP Error 403: Forbidden. The URL provided doesn't allow me to download the image. You can manually download the image and provide the local path to it.")
            except Exception as e:
                sys.exit(f"Error: {str(e)}")
        else:
            image = cv2.imread(args.ImagePath)
        recognized_text = OCR(image)
        if args.TTS=='y' and recognized_text.strip():
            speak_gtts(recognized_text)
        sys.exit()

    if args.imagesFolder:
        if not os.path.exists(args.imagesFolder):
            sys.exit(f"Error: Folder {args.imagesFolder} not found")
        for file in os.listdir(args.imagesFolder):
            image = cv2.imread(os.path.join(args.imagesFolder, file))
            if image is not None:
                recognized_text = OCR(image)
                if recognized_text.strip():
                    print(f"\nRecognized text from {file}:\n{recognized_text}\n")
                    txt_file_name = file + '.txt'
                    with open(os.path.join(args.imagesFolder, txt_file_name), 'w') as f:
                        f.write(recognized_text)
                    if args.TTS=='y':
                        speak_gtts(recognized_text)
                    
        sys.exit()

    elif args.CameraSource:
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