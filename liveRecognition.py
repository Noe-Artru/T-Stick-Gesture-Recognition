## TODO: import libraries, and load pre-trained model
## Read OSC data, and run inference on the model to detect gestures live !

## after testing, latency issue is due to the livePlot function -> needs to be optimized, or something else needs to be used (ascii ?)

from pyo import *
import os
import torch
import keyboard
import numpy as np
import time
import livePlot
import keyboard
import sys
from LSTM_model2 import LSTMClassifier2
import torch.optim as optim

s = Server(sr=48000, buffersize=1024, duplex=1, winhost="asio").boot()
port = 8000
tStickId = "/TStick_195"

input = {
    "raw": {
        "fsr": 0,
        "accl": [],
        "gyro": [],
        "magn": [],
        "capsense": [],
    },
    "instrument": {
        "squeeze": 0,
        "touch": {
            "all": 0,
            "top": 0,
            "middle": 0,
            "bottom": 0,
            "normalised": [],
            "discrete": [],
        },
        "shakexyz": [],
        "jabxyz": [],
        "button": {
            "count": [],
            "tap": [],
            "dtap": [],
            "ttap": []
        },
        "brush": 0,
        "multibrush": [],
        "rub": 0,
        "multirub": [],
    },
    "orientation": [],
    "ypr": [],
    #"battery": 0,
    "battery": {
        "percentage": 0,
        "voltage": 0,
    },
}

# Update input dictionary using OSC data
def fill_dict(input_dict, path, values):
    if len(path) == 1:
        try: input_dict[path[0]] = values
        except Exception as e:
            print("Error:", e)
            print("Path:", path)
    else:
        fill_dict(input_dict[path[0]], path[1:], values)
        

def updateInput(address, *args):
    address = address.split("/")[2:]
    if len(args) == 1:
        args = args[0]
    else:
        args = list(args)

    fill_dict(input, address, args)
    # Only save data once the input dictionnary has been filled, and exactly once per input cycle.
    if(input["raw"]["gyro"] != [] and input["raw"]["magn"] != [] and input["raw"]["accl"] != [] and address[-1] == "accl"):	
        live_data()


def live_data():
    global X, y
    X.append([input["raw"]["fsr"], input["instrument"]["brush"], input["raw"]["accl"][0], input["raw"]["accl"][1], input["raw"]["accl"][2], input["raw"]["gyro"][0], input["raw"]["gyro"][1], input["raw"]["gyro"][2], input["raw"]["magn"][0], input["raw"]["magn"][1], input["raw"]["magn"][2]])
    if(len(X) >= 2*sequence_length):
        X_processed = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
        data = X_processed[-sequence_length:]
        y.append(predict(data).item())
        frame_skip_count = 6 #number of frames skipped between each plot update -> to negate lag, 5 or 6 is needed
        if(len(y) > 100 + frame_skip_count):
            y = y[-100:]
            livePlot.update_plot(y)

        
def predict(data):
    data = torch.tensor(data).float().unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(data)
    return output

gestureNumber = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_sensors = 11
hidden_size = 512
sequence_length = 25
model = LSTMClassifier2(
    num_gestures=1,
    num_sensors=num_sensors,
    hidden_size=hidden_size,
    window_size=sequence_length,
    device=device
)
model.to(device)
state_dict = torch.load("models/gestureRecognition" + str(gestureNumber) + ".pth")
model.load_state_dict(state_dict)

X = []
y = []


scan = OscDataReceive(port=port, address=tStickId + "/*", function=updateInput)
s.start()
s.gui(locals())