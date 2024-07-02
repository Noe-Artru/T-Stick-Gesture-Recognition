from pyo import *
import os
import torch
import keyboard
import numpy as np
import time


# Connection setup
s = Server(sr=48000, buffersize=1024, duplex=1, winhost="asio").boot()
port = 8000
tStickId = "/TStick_195"
is_test = False

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
    if(input["raw"]["gyro"] != [] and input["raw"]["magn"] != [] and input["raw"]["accl"] != [] and address[-1] == "accl"):	
        add_data()

def add_data():
    global X, y
    X.append([input["raw"]["fsr"], input["instrument"]["brush"], input["raw"]["accl"][0], input["raw"]["accl"][1], input["raw"]["accl"][2], input["raw"]["gyro"][0], input["raw"]["gyro"][1], input["raw"]["gyro"][2], input["raw"]["magn"][0], input["raw"]["magn"][1], input["raw"]["magn"][2]])
    if  keyboard.is_pressed(' '):
        y.append(1)
    else:
        y.append(0)


X = []
y = []

scan = OscDataReceive(port=port, address=tStickId + "/*", function=updateInput)

s.start()

while True:
    if keyboard.is_pressed('q'):
        is_test = False
        break
    elif keyboard.is_pressed('p'):
        is_test = True
        break

time.sleep(1) 

# Convert X and y to numpy arrays
X_np = np.array(X)
y_np = np.array(y)

# Save X and y to a file
base_path = 'data/'
i = 0
if(is_test):
    while os.path.exists(base_path + 'X_test' + str(i) + '.npy'):
        i += 1
    np.save(base_path + 'X_test' + str(i) + '.npy', X)
    np.save(base_path + 'y_test' + str(i) + '.npy', y)
else:
    while os.path.exists(base_path + 'X' + str(i) + '.npy'):
        i += 1
    np.save(base_path + 'X' + str(i) + '.npy', X)
    np.save(base_path + 'y' + str(i) + '.npy', y)