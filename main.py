import numpy as np

from pids import PIDS
from pids.utils import create_x0_from_true_model, infer_name_from_acquisition_file

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'cividis'

if __name__ == "__main__":

    # Guess model
    x = np.load("/home/dp4018/data/ultrasound-data/Ultrasound-Vp-sagittal-models/vp_100206.npy")[0]
    x0 = create_x0_from_true_model(x)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(x, vmin=1400, vmax=3000)
    axs[1].imshow(x0, vmin=1400, vmax=3000)
    plt.savefig("models.png")


    # Acquisition and name
    acquisitions_file = "/home/dp4018/data/ultrasound-data/Ultrasound-Vp-sagittal-data/acoustic/data/vp_100206_3shots-Acquisitions.h5"
    name = infer_name_from_acquisition_file(acquisitions_file)
    
    # PIDS instatiantion and processing
    pids = PIDS(acquisitions_file=acquisitions_file, name=name, x0=x0)
    pids.process()
    pids.save(folder_path="./processed/", plot=True)
