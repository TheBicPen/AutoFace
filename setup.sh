#!/usr/bin/env sh

python3 -m venv env
. env/bin/activate
pip install -r requirements.txt
wget https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite
