# Autoface

## Requirements
- Python 3.10 or later
- A webcam

This should run on just about any CPU from the last 10 years.
A dedicated GPU will reduce CPU load.

## Setup

### Linux
1. Read `setup.sh`
2. Run `setup.sh`

### Windows
1. Run the commands in `setup.sh`

Some of the commands may not be available, but there should be equivalent alternatives.

## Usage
1. Run `source env/bin/activate` to enter the python virtual environment
2. Run `python main.py`. This will open a window you can use as a source in any recording or livestreaming application
3. To quit, press 'q'

### OBS
To keep the window at a fixed size in OBS, do the following steps:
1. Right-click the source
2. Go to `Transform` -> `Edit Transform`
3. Change the `Bounding Box Type` from `No bounds` to `Stretch to bounds`
4. Resize the window to whatever size you want

