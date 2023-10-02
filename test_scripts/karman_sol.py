import os, sys, argparse, pickle
from phi.tf.flow import *

params = {}
parser = argparse.ArgumentParser(description='Parameter Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu',            default='0',                                  help='visible GPUs')
parser.add_argument('--model',          default='../data/trained_models/karman/sol/16steps/corrector_model/',     help='path to the corrector model')
parser.add_argument('--stats',      default='../data/trained_models/karman/sol/16steps/dataStats.pickle', help='path to datastats')
parser.add_argument('--output', default = '../data/results/inferences/karman/ato/16steps/', help='output dir')
parser.add_argument('-t', '--simsteps', type = int,  default=200,                                   help='number of frames')
parser.add_argument('-r', '--res',      default=32, type=int,      help='(solver) latent-space (i.e., reduced-space) resolution')
parser.add_argument('-l', '--len',      default=128, type=int,     help='(solver) reference physical dimension')
parser.add_argument('--V0',    default='../data/simulations/karman2d/test_set/',  help='path to hires frames')

sys.argv += ['--' + p for p in params if isinstance(params[p], bool) and params[p]]
pargs = parser.parse_args()
params.update(vars(pargs))

os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']

corrector_model = tf.keras.models.load_model(params['model'])

with open(params['stats'], 'rb') as f: dataStats = pickle.load(f)

dm = dict(y=params['res']*2, x=params['res'], bounds=Box[0:params['len']*2, 0:params['len']],
 extrapolation=extrapolation.combine_sides(y=extrapolation.BOUNDARY, x=extrapolation.ZERO))
 
grid = StaggeredGrid(0, **dm)
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
    v_in = StaggeredGrid(math.stack([v_y.data, v_x.data], channel('vector')), **dm)

    v_in = advect.semi_lagrangian(v_in, v_in, dt)

    return fluid.make_incompressible(v_in, [CYLINDER], Solve('auto', 1e-5, 0, x0=p_in))


def to_feature(pf_grids_in):
    pf_grid = math.stack([math.pad(pf_grids_in[0].vector['x'].values, {'x':(0, 1)} , math.extrapolation.ZERO),
            pf_grids_in[0].vector['y'].y[:-1].values,
            pf_grids_in[1]], channel('channels'))

    return (pf_grid - math.tensor([dataStats['mean'][0], dataStats['mean'][1], dataStats['re.mean']], channel('channels'))
    )/ math.tensor([dataStats['std'][0], dataStats['std'][1], dataStats['re.std']], channel('channels'))
    

def to_staggered(tf_tensor):
    return StaggeredGrid(
        math.stack(
        [
            math.tensor(tf.pad(tf_tensor[0, ..., 1], [(0,1), (0,0)]), math.spatial('y, x')),
            math.tensor(tf_tensor[0, ..., :-1, 0], math.spatial('y, x')),
        ],
        math.channel('vector')),
        **dm)

def eval_model(model_nn, inputs_pf):
    input_tensor = to_feature(inputs_pf)
    math.expand(input_tensor, math.batch('batch', batch=1))
    output_tf = model_nn.predict(input_tensor.native(['batch', 'y', 'x', 'channels']))
    output_tensor = to_staggered(output_tf)*math.tensor([dataStats['std'][1], dataStats['std'][0]],
     channel('vector')) + math.tensor([dataStats['mean'][1], dataStats['mean'][0]], channel('vector'))
    return output_tensor


def write_step(scene, velo, ts):
        scene.write(
            {
            'velo_ls': velo
            },
            frame=ts
        )


for re in [450, 650, 850, 1050, 1200, 1400]:

    scene = Scene.create(params['output'] + f're_{re}/')

    jit_step = math.jit_compile(step)

    vel = phi.field.read(params['V0'] + f'/re_{re}/sim_000000/ds_velo_000000.npz')
    pres = None
    re_tensor = math.ones(CenteredGrid(0, **dm).shape)*re

    for ts in range(1, params['simsteps']):  

        vel, pres = jit_step(vel, pres, re)
        correction = eval_model(corrector_model, [vel, re_tensor])
        vel += correction
        write_step(scene, vel, ts)
