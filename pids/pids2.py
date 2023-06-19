import os
import torch
import numpy as np
from copy import deepcopy

from stride import *
from stride import plotting
from stride.optimisation.optimisers import LocalOptimiser

import matplotlib.pyplot as plt

from utils import infer_name_from_acquisition_file, create_x0_from_true_model


# class RetrieveGradientDescent(LocalOptimiser):
#     """
#     Implementation of a gradient descent class to only retrieve the gradient
#     of the model, i.e. no optimisation updates are performed

#     Parameters
#     ----------
#     variable : Variable
#         Variable to which the optimiser refers.
#     kwargs
#         Extra parameters to be used by the class.

#     """

#     def __init__(self, variable, **kwargs):
#         super().__init__(variable, **kwargs)


#     async def get_grad(self,  grad=None, processed_grad=None, **kwargs):
#         """
#         Apply the optimiser.

#         Parameters
#         ----------
#         grad : Data, optional
#             Gradient to use for the step, defaults to variable gradient.
#         processed_grad : Data, optional
#             Processed gradient to use for the step, defaults to processed variable gradient.
#         kwargs
#             Extra parameters to be used by the method.

#         Returns
#         -------
#         Variable
#             Gradient of the variable.

#         """

#         processed_grad = await self.pre_process(grad=grad,
#                                                 processed_grad=processed_grad,
#                                                 **kwargs)

#         return processed_grad



# TODO:
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


class PIDS:
    def __init__(self, acquisitions_file, geometry_file=None, x0=None, name="acquisition",
                 water_vp=1480., in_mem=False, device="cpu", **kwargs):
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

        :param water_vp [Optional, default 1480.]::
            Acoustic speed of water to consider in the model.
        :type name: ``float``

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
        self.water_vp = water_vp
        self.acquisitions_file = acquisitions_file
        self.geometry_file = geometry_file

        self.acquisitions = None
        self.problem = None
        self.pids = None
        self.vp = None

        self.setup()

        return None
    
    def setup(self):
        if self.acquisitions is None:
            self.acquisitions = Acquisitions(name=self.name)
            self.acquisitions.load(os.path.abspath(self.acquisitions_file))

        if self.x0 is None:
            self.x0 = np.full(self.acquisitions.grid.space.shape, self.water_vp)

        if self.problem is None:
            self.problem = Problem(
                    name=self.name,
                    space=self.acquisitions.grid.space,
                    time=self.acquisitions.grid.time,
                    acquisitions=self.acquisitions
            )

            if self.geometry_file is not None:
                self.problem.geometry.load(os.path.abspath(self.geometry_file))
            else:
                num_locations = len(self.acquisitions.shots)
                self.problem.transducers.default()
                self.problem.geometry.default("elliptical", num_locations)

        if self.vp is None:
            self.vp = ScalarField(
                name='vp', grid=self.problem.grid,
                data=torch.from_numpy(self.x0).to(self.device),
                needs_grad=True
            )
            self.vp.clear_grad()
            self.problem.medium.add(self.vp)
            # self.vp = self.vp.as_parameter()
            # print("****steup", hasattr(self.problem.medium.vp, 'is_proxy'))

            self.problem.medium.add(self.vp)


    def save(self, path=""):
        if not path:
            path = os.path.join(os.getcwd(), self.name + "-pids.npy")
        np.save(path, self.pids)

    async def __process(self, problem, pde, loss_fn, *args, **kwargs):
        logger = mosaic.logger()
        runtime = mosaic.runtime()

        select_shots = kwargs.pop('select_shots',
                                  {'num':len(self.acquisitions.shots)}
                            )

        safe = kwargs.pop('safe', False)

        f_min = kwargs.pop('f_min', None)
        f_max = kwargs.pop('f_max', None)
        process_wavelets = ProcessWavelets.remote(f_min=f_min, f_max=f_max,
                                                len=runtime.num_workers, **kwargs)
        process_traces = ProcessTraces.remote(f_min=f_min, f_max=f_max,
                                            len=runtime.num_workers, **kwargs)

        using_gpu = kwargs.get('platform', 'cpu') == 'nvidia-acc'
        if using_gpu:
            devices = kwargs.pop('devices', None)
            num_gpus = gpu_count() if devices is None else len(devices)
            devices = list(range(num_gpus)) if devices is None else devices

        
        published_args = [runtime.put(each, publish=True) for each in args]
        published_args = await asyncio.gather(*published_args)

        shot_ids = problem.acquisitions.select_shot_ids(**select_shots)

        @runtime.async_for(shot_ids, safe=safe)
        async def loop(worker, shot_id):
            logger.info('\n')
            logger.info('Giving shot %d to %s' % (shot_id, worker.uid))

            sub_problem = problem.sub_problem(shot_id)
            wavelets = sub_problem.shot.wavelets
            observed = sub_problem.shot.observed

            if wavelets is None:
                raise RuntimeError('Shot %d has no wavelet data' % shot_id)

            if observed is None:
                raise RuntimeError('Shot %d has no observed data' % shot_id)

            if using_gpu:
                devito_args = kwargs.get('devito_args', {})
                devito_args['deviceid'] = devices[worker.indices[1] % num_gpus]
                kwargs['devito_args'] = devito_args
   
            wavelets = process_wavelets(wavelets, runtime=worker, **kwargs)
            await wavelets.init_future
            
            modelled = pde(wavelets, *published_args, problem=sub_problem, runtime=worker, **kwargs)
            await modelled.init_future


            traces = process_traces(modelled, observed, runtime=worker, **kwargs)
            await traces.init_future

            loss = await loss_fn(traces.outputs[0], traces.outputs[1],
                             problem=sub_problem, runtime=worker, **kwargs).result()
            
            await loss.adjoint(**kwargs)
            
            logger.info('Retrieved gradient for shot %d' % sub_problem.shot_id)
            
        await asyncio.gather(loop)

        if hasattr(problem.medium.vp, 'is_proxy') and problem.medium.vp.is_proxy:
            await problem.medium.vp.pull(attr='grad')
        grad = problem.medium.vp.grad

        print(np.max(np.abs(grad.data)))

    async def _process(self, runtime):
        platform = None if self.device=="cpu" else "nvidia-acc"
        pde = IsoAcousticDevito.remote(grid=self.problem.grid, len=runtime.num_workers, platform=platform)

        loss_fn = L2DistanceLoss.remote(len=runtime.num_workers)

        await self.__process(self.problem, pde, loss_fn, self.vp)

    def process(self):
        mosaic.run(self._process)

if __name__ == "__main__":

    # Starting model
    x = np.load("/home/dp4018/data/ultrasound-data/Ultrasound-Vp-sagittal-models/vp_100206.npy")[0]
    x0 = create_x0_from_true_model(x)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(x, vmin=1400, vmax=3000)
    axs[1].imshow(x0, vmin=1400, vmax=3000)
    plt.savefig("models.png")


    acquisitions_file = "/home/dp4018/data/ultrasound-data/Ultrasound-Vp-sagittal-data/acoustic/data/vp_100206_3shots-Acquisitions.h5"
    name = infer_name_from_acquisition_file(acquisitions_file)
    pids = PIDS(acquisitions_file=acquisitions_file, name=name)
    pids.process()

    # print(pids.vp.grad)
    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(pids.vp.grad, vmin=1400, vmax=3000)
    # axs[1].imshow(pids.vp.grad, vmin=1400, vmax=3000)
    # plt.savefig("vp-grad.png")

    pids.save()