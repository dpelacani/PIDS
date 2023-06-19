import os
import torch
import numpy as np
from stride import *
import matplotlib.pyplot as plt

# from .utils import infer_name_from_acquisition_file, create_x0_from_true_model

plt.rcParams['image.cmap'] = 'cividis'

async def __process_first_grad(problem, pde, loss_fn, *args, **kwargs):
    logger = mosaic.logger()
    runtime = mosaic.runtime()

    select_shots = kwargs.pop('select_shots',
                                {'num':len(problem.acquisitions.shots)}
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
        
    await loop

    # pull grad to variable
    if hasattr(problem.medium.vp, 'is_proxy') and problem.medium.vp.is_proxy:
        await problem.medium.vp.pull(attr='grad')

    
    logger.info(f'PIDS computed with values in range [{np.min(problem.medium.vp.grad.data)}, {np.max(problem.medium.vp.grad.data)}]')
    logger.info('====================================================================')


async def _process_first_grad(runtime, *args, **kwargs):
    problem = kwargs.pop('problem')
    x0 = kwargs.pop('x0')
    device = kwargs.pop('device', "cpu")

    vp = ScalarField.parameter(
        name='vp', grid=problem.grid,
        data=torch.from_numpy(x0).to(device),
        needs_grad=True
    )
    vp.clear_grad()
    problem.medium.add(vp)

    platform = None if device=="cpu" else "nvidia-acc"
    pde = IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers, platform=platform)

    loss_fn = L2DistanceLoss.remote(len=runtime.num_workers)

    await __process_first_grad(problem, pde, loss_fn, vp)


class PIDS:
    def __init__(self, acquisitions_file, geometry_file=None, x0=None, name="acquisition",
                 water_vp=1480., device="cpu", **kwargs):
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

        :param device:


        """

        self.x0 = x0
        self.name = name
        self.device = device
        self.water_vp = water_vp
        self.acquisitions_file = acquisitions_file
        self.geometry_file = geometry_file

        self.acquisitions = None
        self.problem = None
        self.data = None

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

    def save(self, folder_path="", plot=False):
        assert self.data is not None, "self.process must be called before attempting to save"
        if not folder_path:
            folder_path = os.getcwd()
        path = os.path.join(folder_path, self.name + "-pids")
        np.save(path+".npy", self.data)
        if plot:
            plt.imsave(path+".png", self.data)

    def process(self, *args, **kwargs):
        mosaic.run(_process_first_grad, problem=self.problem, x0=self.x0, *args, **kwargs)
        self.data = self.problem.medium.vp.grad.data