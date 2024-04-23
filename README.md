# read_text_from_camera
Code Python đọc văn bản tiếng Việt từ video camera và chuyển thành giọng nói
## Cài đặt
```git clone https://github.com/brianhuster/read_text_from_camera/```

```cd read_text_from_camera```
```pip install -r requirements.txt```
### Ubuntu
```
sudo apt-get update
sudo apt-get install espeak
sudo apt-get install libgirepository1.0-dev
sudo apt-get install libcairo2-dev
sudo apt-get install -y libgl1-mesa-glx
sudo apt-get install tesseract-ocr-vie
sudo apt-get install -y portaudio19-dev python3-pyaudio
```
### Windows
Install Tesseract [here](https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.3.20231005.exe) and replace ```F:\Tesseract-OCR\tesseract``` in the file main.py with your actual path to Tesseract
### Chạy chương trình
```python3 main.py```

Người dùng ra lệnh, sau đó chương trình sẽ nghe chuyển giọng nói người dùng thành văn bản qua dịch vụ của Google và hiển thị câu lệnh trên Terminal. 

Nếu câu lệnh của người dùng chứa từ "đọc chữ" thì chương trình sẽ chụp hình từ camera rồi chuyển chữ thành văn bản trên terminal, rồi đọc to văn bản qua dịch vụ chuyển văn bản thành giọng nói của Google

## References
[https://github.com/madmaze/pytesseract](https://github.com/madmaze/pytesseract)

[https://github.com/buiquangmanhhp1999/VietnameseOcrCorrection](https://github.com/buiquangmanhhp1999/VietnameseOcrCorrection)
