import os
import torch
import numpy as np
from copy import deepcopy

from stride import *
from stride import plotting


def infer_name_from_acquisition_file(acquisition_file):
    name = os.path.splitext(os.path.basename(acquisition_file))[-2]
    for acq in ["-acquisitions", "acquisitions", "acquisition", "-acquisition"]:
        return name.lower().replace(acq, "")
    if len(name) <= 0:
        return "acquisition"
    return name

# class AcquisitionDataLoader:
#     """
#     An iterator dataloader for .h5 `Stride` acquisitition files that loads data per source.
#     """
#     def __init__(self, acquisition_file, **kwargs):
#         """
#         :param acquisition_file:
#             Acquisition dump file from stride in `h5` format
#         """
#         self.__dict__.update(kwargs)
#         self.acquisition_file = acquisition_file
#         return None
#     def __iter__(self):
#         return None
async def _process(runtime, pids):
    # set up stride problem
    problem = Problem(
            name=pids.name,
            acquisitions=pids.acquisitions_x0
    )

    # acoustic pde
    pde = IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers, platform="nvidia-acc")



class PIDS:
    def __init__(self, acquisitions_file, geometry_file=None, x0=None, name="acquisition",
                 in_mem=False, device="cpu", **kwargs):
        """
        :param acquisitions_file:
            Acquisition dump file from stride in `h5` format.
        :type first: ``str``

        :param geometry_file  [Optional, default None]:
            Geometry dump file from stride in `h5` format. If None, initialises as Stride's default geometry
            with the number of locations equal to the number of sources.
        :type first: ``str``

        :param x0 [Optional, default None]:
            Starting model. If None will be initialised as a homogeneous medium of vp 1500m/s.

        :param name [Optional, default "acquisition"]::
            Name of the Acquistion. Also the name of the PIDS file saved if not specified
            `save` method.
        :type name: ``str``

        :param in_mem [Optional, default False]::
            Whether to load all shots in memory or use a shot-by-shot dataloader
            Not Implemented.
        :type name: ``bool``

        :param device:


        """

        self.x0 = x0
        self.in_mem = in_mem
        self.name = name
        self.device = device
        self.acquisitions_file = acquisitions_file
        self.geometry_file = geometry_file

        self.acquisitions = None
        self.acquisitions_x0 = None
        self.problem = None
        self.pids = None

        self.setup()

        return None
    
    def setup(self):
        if self.acquisitions is None:
            self.load_acquisitions()

        if self.x0 is None:
            self.x0 = np.full(self.acquisitions.grid.space.shape, 1500.)

        if self.acquisitions_x0 is None:
            self.acquisitions_x0 = deepcopy(self.acquisitions)

            for shot_x0 in self.acquisitions_x0.shots: 
                shot_x0.observed.deallocate()  # resets the observed traces

        if self.problem is None:
            self.problem = Problem(
                    name=self.name,
                    space=self.acquisitions.grid.space,
                    time=self.acquisitions.grid.time,
            )

            if self.geometry_file is not None:
                self.problem.geometry.load(os.path.abspath(self.geometry_file))
            else:
                num_locations = len(self.acquisitions.shots)
                self.problem.transducers.default()
                self.problem.geometry.default("elliptical", num_locations)

            self.problem.acquisitions = self.acquisitions_x0

        acquisitions_axes = self.acquisitions.plot(plot=False)
        problem_axes = self.problem.plot(plot=False)
        plotting.show(problem_axes)
        print(problem_axes)


    def load_acquisitions(self):
        self.acquisitions = Acquisitions(name=self.name)
        self.acquisitions.load(os.path.abspath(self.acquisitions_file))

    def save(self, path=""):
        if not path:
            path = os.path.join(os.getcwd(), self.name + "-pids.npy")
        np.save(path, self.pids)

    async def _process(self, runtime):

        # Add medium
        vp = ScalarField(name='vp', grid=self.problem.grid, data=torch.from_numpy(self.x0).to(self.device))
        self.problem.medium.add(vp)

        # acoustic pde
        platform = None if self.device=="cpu" else "nvidia-acc"
        pde = IsoAcousticDevito.remote(grid=self.problem.grid, len=runtime.num_workers, platform=platform)

        await forward(self.problem, pde, vp, dump=False)



    def process(self):
        # mosaic.run(self._process)
        return None

                      

if __name__ == "__main__":
    acquisitions_file = "/home/dp4018/data/ultrasound-data/Ultrasound-Vp-sagittal-data/acoustic/data/vp_100206-Acquisitions.h5"
    name = infer_name_from_acquisition_file(acquisitions_file)
    pids = PIDS(acquisitions_file=acquisitions_file, name=name)
    pids.process()
    pids.save()