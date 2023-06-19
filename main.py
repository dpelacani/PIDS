import os
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

    # Acquisition and name
    acquisitions_file = "/home/dp4018/data/ultrasound-data/Ultrasound-Vp-sagittal-data/acoustic/data/vp_100206_3shots-360-Acquisitions.h5"
    name = infer_name_from_acquisition_file(acquisitions_file)
    
    # PIDS instatiantion
    pids = PIDS(acquisitions_file=acquisitions_file, name=name, x0=x0)

    # Save problem of true model for examples figures
    from stride import *
    vp = ScalarField(
        name='vp', grid=pids.problem.grid,
        data=x,
        needs_grad=True
    )
    pids.problem.medium.add(vp)
    pids.problem.plot(acquisitions=False, cmap="cividis")
    plt.savefig(os.path.join("./figures", "example_x_problem.png"))

    # Save mid shot plot of observed data for examples figures
    mid_shot = pids.problem.acquisitions.shots[int((len(pids.problem.acquisitions.shots) - 1)/2)]
    mid_shot.plot_observed()
    plt.savefig(os.path.join("./figures", "example_data.png"))

    # Process and save PIDS data as .npy; save plots of PIDS and Problem as .png
    pids.process()
    pids.save(folder_path="./processed/", plot=True, cmap="cividis")

