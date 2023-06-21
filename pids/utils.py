import os
import numpy as np

def scale2range(x, range=[-1, 1]):
    return (x - x.min()) * (max(range) - min(range)) / (x.max() - x.min()) + min(range)

def infer_name_from_acquisition_file(acquisition_file):
    name = os.path.splitext(os.path.basename(acquisition_file))[-2]
    for acq in ["-acquisitions", "acquisitions", "acquisition", "-acquisition"]:
        return name.lower().replace(acq, "")
    if len(name) <= 0:
        return "acquisition"
    return name

def create_x0_from_true_model(x, mass_speed=1550, skull_min_speed=None, water_speed=None):
    if water_speed is None:
        water_speed = x[0, 0]
    if skull_min_speed is None:
        skull_min_speed = x.max()*0.7

    x0 = np.full_like(x, mass_speed)
    x0[x==water_speed] = water_speed
    x0[x>= skull_min_speed] = x[x>=skull_min_speed]
    
    return x0