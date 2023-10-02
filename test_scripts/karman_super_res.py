import os, sys, logging, argparse, pickle


log = logging.getLogger()
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

params = {}
parser = argparse.ArgumentParser(description='Parameter Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu',             default='0',                help='visible GPUs')
parser.add_argument('--model',           default=None,               help='path to the trained decoder model (tf)')
parser.add_argument('--stats',           default=None,               help='path to the stats (pickle)')
parser.add_argument('-o', '--output',    default=None,               help='path to output directory')
parser.add_argument('-t', '--simsteps',  default=200, type=int,      help='simulation steps after skipsteps')
parser.add_argument('-b', '--batch',     default=199, type=int,      help='batch size')
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

dm = dict(bounds=Box[0:params['len']*2, 0:params['len']],
 extrapolation=extrapolation.combine_sides(y=extrapolation.BOUNDARY, x=extrapolation.ZERO))

with open(params['stats'], 'rb') as f:
    dataStats = pickle.load(f)
    log.info(dataStats)

def to_feature(pf_grids_in):
    pf_grid = math.stack([math.pad(pf_grids_in.vector['x'].values, {'x':(0, 1)} , math.extrapolation.ZERO),
            pf_grids_in.vector['y'].y[:-1].values], channel('vector'))
        
    return (pf_grid - math.tensor([dataStats['mean'][0], dataStats['mean'][1]], channel('vector'))
        )/ math.tensor([dataStats['std'][0], dataStats['std'][1]], channel('vector'))
        

def to_staggered(tf_tensor):
    return StaggeredGrid(
        math.stack(
        [
            math.tensor(tf.pad(tf_tensor[..., 1], [(0,0), (0,1), (0,0)]), math.batch('batch'), math.spatial('y, x')),
            math.tensor(tf_tensor[..., :-1, 0], math.batch('batch'), math.spatial('y, x')),
        ],
        math.channel('vector')),
        **dm)

def eval_model(model_nn, input_tensor):
    output_tf = model_nn.predict(input_tensor.native(['batch', 'y', 'x', 'vector']))
    output_tensor = to_staggered(output_tf)*math.tensor([dataStats['std'][1], dataStats['std'][0]],
     channel('vector')) + math.tensor([dataStats['mean'][1], dataStats['mean'][0]], channel('vector'))
    return output_tensor


def write_step(scene, dec_velo):
    for i in range(1, params['simsteps']):
        scene.write(
            {
            'velo': StaggeredGrid(
                math.stack(
                    [
                    math.tensor(dec_velo[1][i-1, ...], math.spatial('y, x')),
                    math.tensor(dec_velo[0][i-1, ...], math.spatial('y, x'))
                    ], math.channel('vector')), **dm)
            },
            frame=i
        )

for re in [450, 650, 850, 1050, 1200, 1400]:

    scene = Scene.at(params['output'] + f're_{re}/' + 'sim_000000')
    log.addHandler(logging.FileHandler(os.path.normpath(scene.path)+'/run.log'))

    for timestep in range (1, params['simsteps']):
        velo = phi.field.read(params['V0'] + f're_{re}/' + 'sim_000000/velo_ls_{:06d}.npz'.format(timestep))
        if timestep == 1:
            vel = math.expand(to_feature(velo), math.batch('batch'))
        else:
            vel = math.concat([vel, math.expand(to_feature(velo), math.batch('batch'))], math.batch('batch'))
    
    pf_dec = eval_model(model, vel)

    tf_dec = [pf_dec.vector['x'].values.native(['batch', 'y', 'x']), pf_dec.vector['y'].values.native(['batch', 'y', 'x'])]
    if params['output'] is not None: write_step(scene, tf_dec)
