import subprocess
import time
import sys
import os
from pathlib import Path
import pandas as pd
from subprocess import check_output
import numpy as np

if __name__=='__main__':

    seeds =       [11,   12, 13, 22, 23, 210, 30, 31,  32,  50,  52,  53]
    vehicle_ids = [204, 200, 192,70, 90, 70, 111, 110, 217,500,140, 200]

    for i in [9,10]:
        print("c")

        if i == 9:
            command = ["python", "carla_camera_recorder.py", "--vehicle-id", f"{vehicle_ids[i]}", "--seed",
                       f"{seeds[i]}", "--start_at", "2260"]
        else:
            command = ["python", "carla_camera_recorder.py", "--vehicle-id", f"{vehicle_ids[i]}", "--seed", f"{seeds[i]}"
                       , "--start_at", "402"]

        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, text=True, shell=True, bufsize=1)



        time.sleep(4)
        print('b')

        print(result.stdout)

