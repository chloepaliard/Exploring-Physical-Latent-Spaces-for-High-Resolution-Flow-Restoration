import os, sys, logging, argparse, pickle, distutils.dir_util


log = logging.getLogger()
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

params = {}
parser = argparse.ArgumentParser(description='Parameter Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu',                    default='0',                                                            help='visible GPUs')
parser.add_argument('--encoder_model',          default=None,                                                           help='path to the corrector model')
parser.add_argument('--corrector_model',        default=None,                                                           help='path to the corrector model')
parser.add_argument('--decoder_model',          default=None,                                                           help='path to the corrector model')
parser.add_argument('--stats',                  default=None,                                                           help='path to datastats')
parser.add_argument('--output',                 default=None,                                                           help='output dir')
parser.add_argument('-t', '--simsteps',         default=200, type=int,                                                  help='number of frames')
parser.add_argument('-r', '--res',              default=128, type=int,                                                  help='(solver) latent-space (i.e., reduced-space) resolution')
parser.add_argument('-l', '--len',              default=128, type=int,                                                  help='(solver) reference physical dimension')
parser.add_argument('--V0',                     default=None,                                                            help='path to hires frames')

sys.argv += ['--' + p for p in params if isinstance(params[p], bool) and params[p]]
pargs = parser.parse_args()
params.update(vars(pargs))

os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']

from phi.tf.flow import *

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    log.info('{} Physical GPUs {} Logical GPUs'.format(len(gpus), len(logical_gpus)))

log.info(params)
log.info('tensorflow-{} ({}, {}); keras-{} ({})'.format(tf.__version__, tf.sysconfig.get_include(), tf.sysconfig.get_lib(), keras.__version__, keras.__path__))

encoder_model = tf.keras.models.load_model(params['encoder_model'])
log.info('loaded a trained model: {}'.format(params['encoder_model']))
encoder_model.summary(print_fn=log.info)

corrector_model = tf.keras.models.load_model(params['corrector_model'])
log.info('loaded a trained model: {}'.format(params['corrector_model']))
corrector_model.summary(print_fn=log.info)

decoder_model = tf.keras.models.load_model(params['decoder_model'])
log.info('loaded a trained model: {}'.format(params['decoder_model']))
decoder_model.summary(print_fn=log.info)

with open(params['stats'], 'rb') as f: dataStats = pickle.load(f)

dm = dict(y=pargs.res*2, x=pargs.res, bounds=Box[0:pargs.len*2, 0:pargs.len],
 extrapolation=extrapolation.combine_sides(y=extrapolation.BOUNDARY, x=extrapolation.ZERO))

low_dm = dict(y=pargs.res*2//4, x=pargs.res//4, bounds=Box[0:pargs.len*2, 0:pargs.len],
 extrapolation=extrapolation.combine_sides(y=extrapolation.BOUNDARY, x=extrapolation.ZERO))

grid = StaggeredGrid(0, **low_dm)
shape_v = grid.vector['y'].shape

boundary_c = np.zeros(shape_v.sizes)
boundary_c[0:2, 0:boundary_c.shape[1]-1] = 1.0
boundary_c[0:boundary_c.shape[0], 0:1] = 1.0
boundary_c[0:boundary_c.shape[0], -1:] = 1.0
boundary_c_ = math.tensor(boundary_c, shape_v)
boundary_c_mask = math.tensor(np.copy(boundary_c), shape_v)

CYLINDER = Obstacle(Sphere(center=[32, 64], radius=10))

def step(v_in, p_in, re_in, dt=1.):

    diam = 20
    nu = diam/re_in
    v_in = diffuse.explicit(field=v_in, diffusivity=nu, dt=dt, substeps=1)
    
    v_x = v_in.vector['x']
    v_y = v_in.vector['y']
    v_y = v_y * (1 - boundary_c_mask) + boundary_c_
    v_in = StaggeredGrid(math.stack([v_y.data, v_x.data], channel('vector')), **low_dm)

    v_in = advect.semi_lagrangian(v_in, v_in, dt)

    return fluid.make_incompressible(v_in, [CYLINDER], Solve('auto', 1e-5, 0, x0=p_in))

def to_feature(pf_grids_in):
    if (len(pf_grids_in) > 1):
        pf_grid = math.stack([math.pad(pf_grids_in[0].vector['x'].values, {'x':(0, 1)} , math.extrapolation.ZERO),
                pf_grids_in[0].vector['y'].y[:-1].values,
                pf_grids_in[1]], channel('channels'))

        return (pf_grid - math.tensor([dataStats['mean'][0], dataStats['mean'][1], dataStats['re.mean']], channel('channels'))
    )/ math.tensor([dataStats['std'][0], dataStats['std'][1], dataStats['re.std']], channel('channels'))

    else:
        pf_grid = math.stack([math.pad(pf_grids_in[0].vector['x'].values, {'x':(0, 1)} , math.extrapolation.ZERO),
                pf_grids_in[0].vector['y'].y[:-1].values],
                channel('channels'))

        return (pf_grid - math.tensor([dataStats['mean'][0], dataStats['mean'][1]], channel('channels'))
    )/ math.tensor([dataStats['std'][0], dataStats['std'][1]], channel('channels'))
        

def to_staggered(tf_tensor, domain):
    return StaggeredGrid(
        math.stack(
        [
            math.tensor(tf.pad(tf_tensor[0, ..., 1], [(0,1), (0,0)]), math.spatial('y, x')),
            math.tensor(tf_tensor[0, ..., :-1, 0], math.spatial('y, x')),
        ],
        math.channel('vector')),
        **domain)

def to_staggered_bis(tf_tensor, domain):
    return StaggeredGrid(
        math.stack(
        [
            math.tensor(tf.pad(tf_tensor[..., 1], [(0,0), (0,1), (0,0)]), math.batch('batch'), math.spatial('y, x')),
            math.tensor(tf_tensor[..., :-1, 0], math.batch('batch'), math.spatial('y, x')),
        ],
        math.channel('vector')),
        **domain)

def eval_model(model_nn, inputs_pf, domain):
    input_tensor = to_feature(inputs_pf)
    math.expand(input_tensor, math.batch('batch', batch=1))
    output_tf = model_nn.predict(input_tensor.native(['batch', 'y', 'x', 'channels']))
    output_tensor = to_staggered(output_tf, domain)*math.tensor([dataStats['std'][1], dataStats['std'][0]],
     channel('vector')) + math.tensor([dataStats['mean'][1], dataStats['mean'][0]], channel('vector'))
    return output_tensor


def write_step(scene, dec_velo, velo):
    for i in range(1, params['simsteps']):
        scene.write(
            {
            'velo': StaggeredGrid(
                math.stack(
                    [
                    math.tensor(dec_velo[1][i-1, ...], math.spatial('y, x')),
                    math.tensor(dec_velo[0][i-1, ...], math.spatial('y, x'))
                    ], math.channel('vector')), **dm),
            'velo_ls': StaggeredGrid(
                math.stack(
                    [
                    math.tensor(velo[1][i-1, ...], math.spatial('y, x')),
                    math.tensor(velo[0][i-1, ...], math.spatial('y, x'))
                    ], math.channel('vector')), **low_dm)
            },
            frame=i
        )

jit_step = math.jit_compile(step)

for re in [450, 650, 850, 1050, 1200, 1400]:

    scene = Scene.create(params['output'] + f're_{re}/')
    log.addHandler(logging.FileHandler(os.path.normpath(scene.path)+'/run.log'))

    re_tensor = math.ones(CenteredGrid(0, **dm).shape)*re

    velo = phi.field.read(params['V0'] + f're_{re}/sim_000000/ds_velo_000000.npz')
    velo = eval_model(encoder_model, [velo, re_tensor], low_dm)
    
    pres = None

    for ts in range (1, params['simsteps']):  
        velo, pres = jit_step(velo, pres, re)
        correction = eval_model(corrector_model, [velo, re_tensor], low_dm)
        velo += correction

        if ts == 1:
            vel = math.expand(to_feature(velo), math.batch('batch'))
        else:
            vel = math.concat([vel, math.expand(to_feature(velo), math.batch('batch'))], math.batch('batch'))


    pf_cor = to_staggered_bis(vel, low_dm)

    pf_dec = decoder_model.predict(vel.native(['batch', 'y', 'x', 'vector']))
    pf_dec = to_staggered_bis(pf_dec, dm)*math.tensor([dataStats['std'][1], dataStats['std'][0]],
     channel('vector')) + math.tensor([dataStats['mean'][1], dataStats['mean'][0]], channel('vector'))

    tf_cor = [pf_cor.vector['x'].values.native(['batch', 'y', 'x']), pf_cor.vector['y'].values.native(['batch', 'y', 'x'])]
    tf_dec = [pf_dec.vector['x'].values.native(['batch', 'y', 'x']), pf_dec.vector['y'].values.native(['batch', 'y', 'x'])]
    if params['output'] is not None: write_step(scene, tf_dec, tf_cor)