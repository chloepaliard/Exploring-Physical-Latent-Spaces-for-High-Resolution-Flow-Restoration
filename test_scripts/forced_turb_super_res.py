import os, sys, logging, argparse, pickle

log = logging.getLogger()
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

params = {}
parser = argparse.ArgumentParser(description='Parameter Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu',             default='0',                help='visible GPUs')
parser.add_argument('--model',           default=None,               help='path to the trained decoder model (tf)')
parser.add_argument('--stats',           default=None,               help='path to the dataset stats (pickle)')
parser.add_argument('-o', '--output',    default=None,               help='path to output directory')
parser.add_argument('-n', '--nsims',     default=10, type=int,       help='test simulations')
parser.add_argument('-t', '--simsteps',  default=200, type=int,      help='simulation steps after skipsteps')
parser.add_argument('-b', '--sbatch',    default=100, type=int,      help='batch size')
parser.add_argument('-l', '--len',       default=128, type=int,      help='reference physical dimension')
parser.add_argument('--V0',              default=None,               help='initial high-res state')
sys.argv += ['--' + p for p in params if isinstance(params[p], bool) and params[p]]
pargs = parser.parse_args()
params.update(vars(pargs))

assert params['output'] is not None, 'No output path is given.'
assert params['model'] is not None, 'No decoder model is given.'
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

model = tf.keras.models.load_model(params['model'], compile=False)
model.summary(print_fn=log.info)

with open(params['stats'], 'rb') as f:
    dataStats = pickle.load(f)
    log.info(dataStats)
    dataMean = math.tensor([dataStats['mean'][1], dataStats['mean'][0],], channel('vector')) # [v, u]
    dataStd  = math.tensor([dataStats['std'][1],  dataStats['std'][0],],  channel('vector'))

def to_feature(pf_vel_grid, data_mean, data_std):
    return (pf_vel_grid.values - data_mean)/data_std

def to_centered(tf_tensor):
    return CenteredGrid(
        math.tensor(tf_tensor, math.batch('sbatch'), math.spatial('y, x'), math.channel('vector')), # drop the batch dim
        bounds=Box[0:params['len'], 0:params['len']], extrapolation=extrapolation.PERIODIC
    )

def eval_model(model_nn, input_pf, data_mean, data_std):
    model_input = to_feature(input_pf, data_mean, data_std)
    output_tf = model_nn.predict(model_input.native(['sbatch', 'y', 'x', 'vector']))
    return to_centered(output_tf)*data_std + data_mean

def write_step(scene, dec_velo):
    for i in range(1, params['simsteps']):
        scene.write(
            data = {
                'velo': CenteredGrid(math.tensor(dec_velo[i-1, ...], math.spatial('y, x'), math.channel('vector')), 
                bounds=Box[0:params['len'], 0:params['len']], extrapolation=extrapolation.PERIODIC)
            },
            frame=i
        )

for sim_idx in range(params['nsims']):

    scene = Scene.at(params['output'] + '/sim_{:06d}/'.format(sim_idx))
    log.addHandler(logging.FileHandler(os.path.normpath(scene.path)+'/run.log'))

    dm = dict(bounds=Box[0:params['len'], 0:params['len']])

    for timestep in range (1, params['simsteps']):
        velo = phi.field.read(params['V0'] + 'sim_{:06d}/velo_ls_{:06d}.npz'.format(sim_idx, timestep)).values.numpy(['y', 'x', 'vector'])
        if timestep == 1:
            vel = np.expand_dims(velo, axis=0)
        else:
            new_vel = velo
            vel = np.concatenate([vel, np.expand_dims(new_vel, axis=0)], axis=0)

    pf_vel = CenteredGrid(math.tensor(vel, math.batch('sbatch'), math.spatial('y, x'), math.channel('vector')),
     extrapolation=extrapolation.PERIODIC, bounds=dm['bounds'])
    
    pf_dec = eval_model(model, pf_vel, dataMean, dataStd)

    tf_dec = pf_dec.values.native(['sbatch', 'y', 'x', 'vector'])
    if params['output'] is not None: write_step(scene, tf_dec)