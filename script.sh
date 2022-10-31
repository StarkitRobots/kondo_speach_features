#!/usr/bin/ bash



pip install SpeechRecognition
pip install pyaudio
pip install editdistance
pip install gTTS
pip install sklearn
pip install nltk

sudo apt install portaudio19-dev python3-pyaudio
sudo apt-get install portaudio19-dev python-pyaudio python3-pyaudio
sudo apt update && sudo apt install espeak ffmpeg libespeak1 for offline text_to_voice convert
pip install yandex-music --upgrade