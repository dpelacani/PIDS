import os
import numpy as np

from pids import PIDS
from pids.utils import create_x0_from_true_model, infer_name_from_acquisition_file

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'cividis'

if __name__ == "__main__":

    # Model for backpropagation -- here we use a version of the true model that only contains the skull
    x = np.load("/home/dp4018/data/ultrasound-data/Ultrasound-Vp-transverse-models/vp_204622.npy")
    x0 = create_x0_from_true_model(x)


    # Stride dump files and name
    acquisitions_file = "/home/dp4018/data/ultrasound-data/Ultrasound-Vp-transverse-data/acoustic/data/vp_204622_36shots-512resize-highf-Acquisitions.h5"
    geometry_file = "/home/dp4018/data/ultrasound-data/Ultrasound-Vp-transverse-data/acoustic/data/vp_204622_36shots-512resize-highf-Geometry.h5"
    transducers_file = "/home/dp4018/data/ultrasound-data/Ultrasound-Vp-transverse-data/acoustic/data/vp_204622_36shots-512resize-highf-Transducers.h5"
    name = infer_name_from_acquisition_file(acquisitions_file)
    
    # PIDS instatiantion
    pids = PIDS(acquisitions_file=acquisitions_file, geometry_file=geometry_file, transducers_file=transducers_file, name=name, x0=x0)

    # Process and save PIDS data as .npy; save plots of PIDS and Problem as .png
    pids.process()
    pids.save(folder_path="./processed/", plot=True, cmap="cividis")

