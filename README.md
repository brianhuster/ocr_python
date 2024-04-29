﻿# OCR_python
This python script can extract text from image (local or URL), and camera
## Install
```shell
git clone https://github.com/brianhuster/read_text_from_camera/
cd read_text_from_camera
pip install -r requirements.txt
```

## Usage
```shell
python3 script.py [-h] [-i IMAGEPATH] [-c CAMERASOURCE] [-d {easyocr,tesseract}] [-r {easyocr,easy_ocr,vietocr}] [-l LANGUAGE] [-t {y,n}]
```

Arguments:
- `-h`, `--help`: For help
- `-i IMAGEPATH`, `--ImagePath IMAGEPATH`: Local path or URL to an image
- `-c CAMERASOURCE`, `--CameraSource CAMERASOURCE`: Index or URL of the video camera (default: 0)
- `-d {easyocr,tesseract}`, `--detector {easyocr,tesseract}`: Detector to use (default: easyocr)
- `-r {easyocr,easy_ocr,vietocr}`, `--recognizer {easyocr,easy_ocr,vietocr}`: Recognizer to use (default: easyocr)
- `-l LANGUAGE`, `--language LANGUAGE`: 2-letter code of language to use for OCR (default: vi)
- `-t {y,n}`, `--textCorrector {y,n}`: Use text corrector or not (y/n) for Vietnamese text (default: n)

### Examples

1. Recognize text from an image file:
```shell
python3 script.py -i image.jpg
```
2. Recognize text from a camera feed:
```shell
python script.py -c 0
```
or 
```shell
python3 script.py
```
Say "đọc chữ" or "read it" to trigger text recognition and speech synthesis.

While the camera is running, press ```q``` to turn it off

### Demo video
[![Video](https://i9.ytimg.com/vi/v9YPcHzfTyk/mqdefault.jpg?sqp=CNjYrrEG-oaymwEmCMACELQB8quKqQMa8AEB-AH-CYAC0AWKAgwIABABGGUgQyhCMA8=&rs=AOn4CLCxICP7zv-lwkr3xoMfB8t1JQ5alw)](https://www.youtube.com/watch?v=v9YPcHzfTyk)

## References
[https://github.com/madmaze/pytesseract](https://github.com/madmaze/pytesseract)

[https://github.com/pbcquoc/vietocr](https://github.com/pbcquoc/vietocr)

[https://github.com/JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR)

[https://github.com/buiquangmanhhp1999/VietnameseOcrCorrection](https://github.com/buiquangmanhhp1999/VietnameseOcrCorrection)