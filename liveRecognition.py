## TODO: import libraries, and load pre-trained model
## Read OSC data, and run inference on the model to detect gestures live !

## after testing, latency issue is due to the livePlot function -> needs to be optimized, or something else needs to be used (ascii ?)

from pyo import *
import os
import torch
import keyboard
import numpy as np
import time
import plotLive
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
        
start = time.time()
def updateInput(address, *args):
    global start
    address = address.split("/")[2:]
    if len(args) == 1:
        args = args[0]
    else:
        args = list(args)

    fill_dict(input, address, args)
    # Only save data once the input dictionnary has been filled, and exactly once per input cycle.
    if(input["raw"]["gyro"] != [] and input["raw"]["magn"] != [] and input["raw"]["accl"] != [] and address[-1] == "accl"):	
        live_data() # on average this gets called every 20 miliseconds



#Latency study: live data needs to run in less than 20 miliseconds.
# To reduce data manipulation latency (0.1 milisecond on average), we never store more than 4*sequence_length data points
# # If we want to plot, is adds 7 miliseconds of latency on average (using dynamic plotting techniques)
# Model inference takes 4 miliseconds on average for 512 hidden_size, 25 sequence_length, 7 input feature
def live_data():
    global X, y, means, stds
    X.append([input["raw"]["fsr"], input["raw"]["accl"][0], input["raw"]["accl"][1], input["raw"]["accl"][2], input["raw"]["gyro"][0], input["raw"]["gyro"][1], input["raw"]["gyro"][2]])
    if(len(X) >= 5*sequence_length): #reduce latency by partially storing X
        X = X[-sequence_length:]
    if(len(X) >= sequence_length):
        data = (X[-sequence_length:]-means) / (stds + 1e-8)
        y.append(predict(data).item())
        frame_skip_count = 3 #number of frames skipped between each plot update -> to negate lag
        if(len(y) > 100 + frame_skip_count):
            y = y[-100:]
            plotLive.update_plot(y) # takes around 10 miliseconds to update plot

def predict(data):
    data = torch.tensor(data).float().unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(data) # model inference takes 3 miliseconds for larger sized model
    return output

gestureNumber = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_sensors = 7
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
model.load_state_dict(state_dict["model_state_dict"])
means = state_dict["means"]
stds = state_dict["stds"]

X = []
y = []

scan = OscDataReceive(port=port, address=tStickId + "/*", function=updateInput)
s.start()
s.gui(locals())