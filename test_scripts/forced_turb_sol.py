import os, sys, logging, argparse, pickle
from timeit import default_timer as timer

log = logging.getLogger()
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

params = {}
parser = argparse.ArgumentParser(description='Parameter Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu',             default='0',                help='visible GPUs')
parser.add_argument('--model',           default=None,               help='path to the trained model (tf)')
parser.add_argument('--stats',           default=None,               help='path to the dataset stats (pickle)')
parser.add_argument('-o', '--output',    default=None,               help='path to output directory')
parser.add_argument('-n', '--nsims',     default=10, type=int,       help='test simulations')
parser.add_argument('-t', '--simsteps',  default=200, type=int,      help='simulation steps after skipsteps')
parser.add_argument('-l', '--len',       default=128, type=int,      help='reference physical dimension')
parser.add_argument('--nu',              default=0.1, type=float,    help='diffusion coefficient')
parser.add_argument('--dt',              default=0.2, type=float,    help='simulation time step size')
parser.add_argument('--V0',              default=None,               help='initial high-res state')
sys.argv += ['--' + p for p in params if isinstance(params[p], bool) and params[p]]
pargs = parser.parse_args()
params.update(vars(pargs))

assert params['output'] is not None, 'No output path is given.'
assert params['model'] is not None, 'No model is given.'
assert params['stats'] is not None, 'No stats path is given.'

os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']

from phi.tf.flow import *

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    log.info('{} Physical GPUs {} Logical GPUs'.format(len(gpus), len(logical_gpus)))

log.info(params)
log.info('tensorflow-{} ({}, {}); keras-{} ({})'.format(tf.__version__, tf.sysconfig.get_include(), tf.sysconfig.get_lib(), keras.__version__, keras.__path__))

model = tf.keras.models.load_model(params['model'])
log.info('loaded a trained model: {}'.format(params['model']))
model.summary(print_fn=log.info)

with open(params['stats'], 'rb') as f:
    dataStats = pickle.load(f)
    log.info(dataStats)
    dataMean = math.tensor([dataStats['mean'][1], dataStats['mean'][0],], channel('vector')) # [v, u]
    dataStd  = math.tensor([dataStats['std'][1],  dataStats['std'][0],],  channel('vector'))

class SolverTurbulence():
    def __init__(self, viscosity=0.1, substeps=1):
        self.viscosity = viscosity
        self.substeps = substeps
        self.appTime = 0

    def step(self, v_in, p_in, f_in, dt=1.0):
        velocity = advect.semi_lagrangian(field=v_in, velocity=v_in, dt=dt)
        velocity = velocity + dt*f_in
        velocity = diffuse.explicit(field=velocity, diffusivity=self.viscosity, dt=dt, substeps=self.substeps)
        velocity, pressure = fluid.make_incompressible(velocity, (), Solve('auto', 1e-5, 1e-5, x0=p_in))
        self.appTime += dt
        return velocity, pressure

def to_feature(pf_vel_grid, data_mean, data_std):
    return (pf_vel_grid.values - data_mean)/data_std

def to_centered(tf_tensor):
    assert tf_tensor.shape[0]==1, 'the batch size is not one.'
    return CenteredGrid(
        math.tensor(tf_tensor[0, ...], math.spatial('y, x'), math.channel('vector')), # drop the batch dim
        bounds=Box[0:params['len'], 0:params['len']], extrapolation=extrapolation.PERIODIC
    )

def eval_model(model_nn, input_pf, data_mean, data_std):
    input_model = to_feature(input_pf, data_mean, data_std)
    output_tf = model_nn.predict(input_model.native(['batch', 'y', 'x', 'vector']))
    return to_centered(output_tf)*data_std + data_mean

def write_step(scene, velo, frame_i):
    scene.write(
        data = {
            'velo_ls': velo
        },
        frame=frame_i
    )

mySolver = SolverTurbulence(viscosity=params['nu'])
jit_step = math.jit_compile(mySolver.step)

for sim_idx in range(params['nsims']):

    scene = Scene.create(params['output'])
    log.addHandler(logging.FileHandler(os.path.normpath(scene.path)+'/run.log'))

    pres = None

    if params['V0']:               
        log.info('loading initial velocity field: {}'.format(params['V0']))
        vel = phi.field.read(params['V0'] + '/sim_{:06d}/ds_velo_000000.npz'.format(sim_idx))
   
    for i in range(1, params['simsteps']):
        log.info('Step {:06d}'.format(i))

        fext = phi.field.read(params['V0'] + '/sim_{:06d}/ds_f_ext_{:06d}.npz'.format(sim_idx, i-1))

        vel, pres = jit_step(vel, pres, fext, params['dt'])

        correction = eval_model(model, vel, dataMean, dataStd)
        vel = vel + correction
        if params['output'] is not None: write_step(scene, vel, i)