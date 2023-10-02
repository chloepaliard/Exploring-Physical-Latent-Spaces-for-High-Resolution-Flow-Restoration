import os, sys, logging, argparse, pickle, distutils.dir_util

log = logging.getLogger()
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

params = {}
parser = argparse.ArgumentParser(description='Parameter Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu',             default='0',                help='visible GPUs')
parser.add_argument('--enc_model',       default=None,               help='path to the trained encoder model (tf)')
parser.add_argument('--cor_model',       default=None,               help='path to the trained corrector model (tf)')
parser.add_argument('--dec_model',       default=None,               help='path to the trained decoder model (tf)')
parser.add_argument('--stats',           default=None,               help='path to the dataset stats (pickle)')
parser.add_argument('-o', '--output',    default=None,               help='path to output directory')
parser.add_argument('-n', '--nsims',     default=10, type=int,       help='test simulations')
parser.add_argument('-t', '--simsteps',  default=200, type=int,      help='simulation steps after skipsteps')
parser.add_argument('-b', '--sbatch',    default=100, type=int,      help='batch size')
parser.add_argument('-l', '--len',       default=128, type=int,      help='reference physical dimension')
parser.add_argument('--nu',              default=0.1, type=float,    help='diffusion coefficient')
parser.add_argument('--dt',              default=1.0, type=float,    help='simulation time step size')
parser.add_argument('--V0',              default=None,               help='initial high-res state')
sys.argv += ['--' + p for p in params if isinstance(params[p], bool) and params[p]]
pargs = parser.parse_args()
params.update(vars(pargs))

assert params['enc_model'] is not None, 'No encoder model is given.'
assert params['cor_model'] is not None, 'No corrector model is given.'
assert params['dec_model'] is not None, 'No decoder model is given.'
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

enc_model = tf.keras.models.load_model(params['enc_model'])
log.info('loaded a trained encoder model: {}'.format(params['enc_model']))
enc_model.summary(print_fn=log.info)

cor_model = tf.keras.models.load_model(params['cor_model'])
log.info('loaded a trained corrector model: {}'.format(params['cor_model']))
cor_model.summary(print_fn=log.info)

dec_model = tf.keras.models.load_model(params['dec_model'])
log.info('loaded a trained decoder model: {}'.format(params['dec_model']))
dec_model.summary(print_fn=log.info)

with open(params['stats'], 'rb') as f:
    dataStats = pickle.load(f)
    log.info(dataStats)
    dataMean = math.tensor([dataStats['mean'][1], dataStats['mean'][0],], channel('vector')) # [v, u]
    dataStd  = math.tensor([dataStats['std'][1],  dataStats['std'][0],],  channel('vector'))
    fextMean = math.tensor([dataStats['mean'][3], dataStats['mean'][2],], channel('vector')) # [fy, fx]
    fextStd  = math.tensor([dataStats['std'][3],  dataStats['std'][2],],  channel('vector'))

class SolverTurbulence():
    def __init__(self, viscosity=0.1, substeps=1, dt=1.0):
        self.viscosity = viscosity
        self.substeps = substeps
        self.dt = dt
        self.appTime = 0

    def step(self, v_in, p_in, f_in):
        velocity = advect.semi_lagrangian(field=v_in, velocity=v_in, dt=self.dt)
        velocity = velocity + self.dt*f_in
        velocity = diffuse.explicit(field=velocity, diffusivity=self.viscosity, dt=self.dt, substeps=self.substeps)
        velocity, pressure = fluid.make_incompressible(velocity, (), Solve('auto', 1e-5, 1e-5, x0=p_in))
        self.appTime += self.dt
        return velocity, pressure

def to_feature(pf_vel_grid, data_mean, data_std):
    return (pf_vel_grid.values - data_mean)/data_std

def to_centered(tf_tensor):
    assert tf_tensor.shape[0]==1, 'the batch size is not one.'
    return CenteredGrid(
        math.tensor(tf_tensor[0, ...], math.spatial('y, x'), math.channel('vector')), # drop the batch dim
        bounds=dm['bounds'], extrapolation=extrapolation.PERIODIC
    )

def eval_model(model_nn, input_pf, data_mean, data_std):
    input_model = to_feature(input_pf, data_mean, data_std)
    output_tf = model_nn.predict(input_model.native(['batch', 'y', 'x', 'vector']))
    return to_centered(output_tf)*data_std + data_mean

def to_centered_batch(tf_tensor):
    return CenteredGrid(
        math.tensor(tf_tensor, math.batch('sbatch'), math.spatial('y, x'), math.channel('vector')),
        bounds=dm['bounds'], extrapolation=extrapolation.PERIODIC
    )

def eval_model_batch(model_nn, input_pf, data_mean, data_std):
    model_input = to_feature(input_pf, data_mean, data_std)
    output_tf = model_nn.predict(model_input.native(['sbatch', 'y', 'x', 'vector']))
    return to_centered_batch(output_tf)*data_std + data_mean


def write_step(scene, dec_velo, cor_velo):
    for i in range(params['simsteps'] - 1):
        scene.write(
            data = {
                'velo': CenteredGrid(math.tensor(dec_velo[i, ...], math.spatial('y, x'), math.channel('vector')), 
                bounds=dm['bounds'], extrapolation=extrapolation.PERIODIC),
                'velo_ls': CenteredGrid(math.tensor(cor_velo[i, ...], math.spatial('y, x'), math.channel('vector')), 
                bounds=dm['bounds'], extrapolation=extrapolation.PERIODIC),
            },
            frame=i+1
        )

for sim_idx in range(params['nsims']):

    scene = Scene.create(params['output'])
    log.addHandler(logging.FileHandler(os.path.normpath(scene.path)+'/run.log'))

    mySolver = SolverTurbulence(viscosity=params['nu'], dt=params['dt'])
    jit_step = math.jit_compile(mySolver.step)

    dm = dict(bounds=Box[0:params['len'], 0:params['len']])
    vel0 = phi.field.read(params['V0'] + '/sim_{:06d}/ds_velo_000000.npz'.format(sim_idx))

    pf_cor = []
    pf_dec = []
    fext = []

    pres = None

    for i in range(params['simsteps'] - 1):
        fext_gt = phi.field.read(params['V0'] + '/sim_{:06d}/ds_f_ext_{:06d}.npz'.format(sim_idx, i))
        fext.append(eval_model(enc_model, fext_gt, fextMean, fextStd))

    vel = eval_model(enc_model, vel0, dataMean, dataStd)

    for i in range(1, params['simsteps']):
        log.info('Step {:06d}'.format(i))

        vel, pres = jit_step(vel, pres, fext[i-1])

        correction = eval_model(cor_model, vel, dataMean, dataStd)
        vel += correction

        if i==1:
            corrected =  np.expand_dims(vel.values.numpy(['y', 'x', 'vector']), axis=0)
        else:
            corrected = np.concatenate([corrected,  np.expand_dims(vel.values.numpy(['y', 'x', 'vector']), axis=0)], axis=0)

    pf_cor = CenteredGrid(math.tensor(corrected, math.batch('sbatch'), math.spatial('y, x'), math.channel('vector')),
     extrapolation=extrapolation.PERIODIC, bounds=dm['bounds'])

    pf_dec = eval_model_batch(dec_model, pf_cor, dataMean, dataStd)

    tf_cor = pf_cor.values.native(['sbatch', 'y', 'x', 'vector'])
    tf_dec = pf_dec.values.native(['sbatch', 'y', 'x', 'vector'])
    if params['output'] is not None: write_step(scene, tf_dec, tf_cor)