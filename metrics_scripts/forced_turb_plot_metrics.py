import argparse, glob
import numpy as np
import matplotlib.pyplot as plt
from phi.flow import *

params = {}
parser = argparse.ArgumentParser(description='Parameter Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mae',                action='store_true',               help='compute MAEs')
parser.add_argument('--mse',                action='store_true',               help='compute MSEs')
parser.add_argument('--reduced',            action='store_true',               help='compute reduced space distance to lerp')
parser.add_argument('-s', '--nSims',        default=5, type=int,               help='number of simulations')
parser.add_argument('-t', '--timesteps',    default=199, type=int,             help='number of timesteps')
parser.add_argument('-r', '--res',          default=128, type=int,             help='resolution')
parser.add_argument('-l', '--len',          default=128, type=int,             help='domain length')

pargs = parser.parse_args()

assert (pargs.mae or pargs.mse or pargs.reduced), "Must plot at least one metric (add arg mae and/or mse and/or reduced)"

# Plots parameters
plt.rcParams.update(
    {
        'grid.alpha': 0.3,
        'grid.color': 'gray',
        'grid.linestyle': 'solid',
        'grid.linewidth': 0.5,
        'axes.labelsize': 'xx-large',
        'xtick.labelsize':'x-large',
        'ytick.labelsize':'x-large',
		'axes.titlesize': 'xx-large',	
    }
)

colors = ['darkgray', 'navy', 'brown', 'orange']

if pargs.mae or pargs.mse:
    labels = ['baseline', 'Dil-ResNet + SR', 'SOL + SR', 'ATO (ours)']
    X_bars = ['baseline', 'Dil-ResNet\n+ SR', 'SOL\n+ SR', 'ATO\n(ours)']

if pargs.reduced:
    labels_reduced = ['baseline', 'Dil-ResNet', 'SOL', 'ATO (ours)']
    X_bars_reduced = ['baseline', 'Dil-ResNet', 'SOL', 'ATO\n(ours)']

X = np.linspace(1, pargs.timesteps, pargs.timesteps)

# Vorticity from u and v
def curlCen(u, v):
    cen = np.zeros(shape=u.shape)

    dv_0 = v[:, 1:2] - v[:, 0:1] #boundaries
    dv_middle = v[:, 2:] - v[:, :v.shape[1] - 2]
    dv_end = v[:, -1:] - v[:, -2:-1] #boundaries
    dv = np.concatenate((dv_0, dv_middle, dv_end), axis=1)

    du_0 = u[1:2, :] - u[0:1, :] #boundaries
    du_middle = u[2:, :] - u[:u.shape[0] - 2, :]
    du_end = u[-1:, :] - u[-2:-1, :] #boundaries
    du = np.concatenate((du_0, du_middle, du_end), axis=0)

    cen = 0.5 * (dv - du)
    return cen


ref_dir = 'data/forced_turb/'

dir = ['data/forced_turb/baseline/']
dir.append('inferences/forced_turb/dilresnet/')
dir.append('inferences/forced_turb/sol/')
dir.append('inferences/forced_turb/ato/')

vel_mae  = np.zeros((pargs.nSims, len(dir), pargs.timesteps)) 
vort_mae = np.zeros((pargs.nSims, len(dir), pargs.timesteps))
vel_mse  = np.zeros((pargs.nSims, len(dir), pargs.timesteps)) 
vort_mse = np.zeros((pargs.nSims, len(dir), pargs.timesteps))
reduced_mae = np.zeros((pargs.nSims, len(dir), pargs.timesteps))

for sim_idx in range(pargs.nSims):

    ref_path_list  = sorted(glob.glob(ref_dir + 'sim_{:06d}/velo_0*.npz'.format(sim_idx)))[1:200]
    ref_ds_path_list  = sorted(glob.glob(ref_dir + 'sim_{:06d}/ds_velo_0*.npz'.format(sim_idx)))[1:200]
    base_path_list = sorted(glob.glob(dir[0] + 'sim_{:06d}/velo_0*.npz'.format(sim_idx)))[1:200]
    
    for ts in range(pargs.timesteps):

        ref_vel = phi.field.read(ref_path_list[ts]).values.numpy(['y', 'x', 'vector'])
        # For vorticity calculation
        v_ref = ref_vel[..., 0]
        u_ref = ref_vel[..., 1] 

        base_vel = phi.field.read(base_path_list[ts]).at(CenteredGrid((0, 0), y=pargs.res, x=pargs.res,
         bounds=Box[0:pargs.len, 0:pargs.len])).values.numpy(['y', 'x', 'vector'])
        # For vorticity calculation
        v = base_vel[..., 0]
        u = base_vel[..., 1] 

        ref_ds_vel = phi.field.read(ref_ds_path_list[ts]).values.numpy(['y', 'x', 'vector'])
        base_ds_vel = phi.field.read(base_path_list[ts]).values.numpy(['y', 'x', 'vector'])
        
        if pargs.mae:
            vel_mae[sim_idx][0][ts] = np.mean(np.abs(base_vel - ref_vel))
            vort_mae[sim_idx][0][ts] = np.mean(np.abs(curlCen(u, v) - curlCen(u_ref, v_ref)))

        if pargs.mse:
            vel_mse[sim_idx][0][ts] = np.mean(np.square(base_vel - ref_vel))
            vort_mse[sim_idx][0][ts] = np.mean(np.square(curlCen(u, v) - curlCen(u_ref, v_ref)))

        if pargs.reduced:
            reduced_mae[sim_idx][0][ts] = np.mean(np.abs(base_ds_vel - ref_ds_vel))

        for model_idx in range(1, len(dir)):

            vel_path  = sorted(glob.glob(dir[model_idx] + 'sim_{:06d}/velo_0*.npz'.format(sim_idx)))[ts]
            vel = phi.field.read(vel_path).values.numpy(['y', 'x', 'vector'])
            
            if pargs.reduced:
                cor_vel_path  = sorted(glob.glob(dir[model_idx] + 'sim_{:06d}/velo_ls_0*.npz'.format(sim_idx)))[ts]
                cor_vel = phi.field.read(cor_vel_path).values.numpy(['y', 'x', 'vector'])

            # For vorticity calculation
            v = vel[..., 0]
            u = vel[..., 1] 
        
            if pargs.mae:
                vel_mae[sim_idx][model_idx][ts] = np.mean(np.abs(vel - ref_vel))
                vort_mae[sim_idx][model_idx][ts] = np.mean(np.abs(curlCen(u, v) - curlCen(u_ref, v_ref)))

            if pargs.mse:
                vel_mse[sim_idx][model_idx][ts] = np.mean(np.square(vel - ref_vel))
                vort_mse[sim_idx][model_idx][ts] = np.mean(np.square(curlCen(u, v) - curlCen(u_ref, v_ref)))

            if pargs.reduced:
                reduced_mae[sim_idx][model_idx][ts] = np.mean(np.abs(cor_vel - ref_ds_vel))


if pargs.mae:
    vel_mae_time = np.mean(vel_mae, axis=0)
    vort_mae_time = np.mean(vort_mae, axis=0)

if pargs.mse:
    vel_mse_time = np.mean(vel_mse, axis=0)
    vort_mse_time = np.mean(vort_mse, axis=0)

if pargs.reduced:
    reduced_mae_time = np.mean(reduced_mae, axis=0)


# Plot errors along time
if pargs.mae:
    for j in range(len(dir)):
        plt.plot(X, vel_mae_time[j], color = colors[j], linewidth=1.0, label = labels[j])

    plt.xlabel("Timestep")
    plt.ylabel('MAE velocity')
    plt.grid(axis='both')
    plt.legend(ncol=1)
    plt.savefig('forced_turb_velo_mae_time.pdf')
    plt.close()

if pargs.reduced:
    for j in range(len(dir)):
        plt.plot(X, reduced_mae_time[j], color = colors[j], linewidth=1.0, label = labels_reduced[j])

    plt.xlabel("Timestep")
    plt.ylabel('MAE lerp(ref)')
    plt.grid(axis='both')
    plt.legend(ncol=1)
    plt.savefig('forced_turb_reduced_time.pdf')
    plt.close()


if pargs.mae:

    vel_mae_std = np.std(np.mean(vel_mae, axis=-1), axis=0)
    vort_mae_std = np.std(np.mean(vort_mae, axis=-1), axis=0)

    vel_mae_mean = np.mean(vel_mae_time, axis=-1)
    vort_mae_mean = np.mean(vort_mae_time, axis=-1)

    Y_bars_vel_mae  = [vel_mae_mean[i] for i in range(len(dir))]
    Y_bars_vort_mae = [vort_mae_mean[i] for i in range(len(dir))]

    vel_mae_mean_over_time = np.mean(vel_mae, axis=-1)

    vel_improv_mae_over_time = np.zeros((pargs.nSims, len(dir))) 
    for i in range(1, len(dir)):
        vel_improv_mae_over_time[:, i-1] = (vel_mae_mean_over_time[:, 0] - vel_mae_mean_over_time[:, i])/vel_mae_mean_over_time[:, 0]

    vel_improv_mae_std = np.std(vel_improv_mae_over_time, axis=0)
    vel_improv_mae_mean = np.mean(vel_improv_mae_over_time, axis=0)
    Y_bars_vel_improv_mae  = [vel_improv_mae_mean[i] for i in range(len(dir)-1)]

    vort_mae_mean_over_time = np.mean(vort_mae, axis=-1)
    
    vort_improv_mae_over_time = np.zeros((pargs.nSims, len(dir))) 
    for i in range(1, len(dir)):
        vort_improv_mae_over_time[:, i-1] = (vort_mae_mean_over_time[:, 0] - vort_mae_mean_over_time[:, i])/vort_mae_mean_over_time[:, 0]  

    vort_improv_mae_std = np.std(vort_improv_mae_over_time, axis=0)
    vort_improv_mae_mean = np.mean(vort_improv_mae_over_time, axis=0)
    Y_bars_vort_improv_mae  = [vort_improv_mae_mean[i] for i in range(len(dir)-1)]

if pargs.mse:
    vel_mse_std = np.std(np.mean(vel_mse, axis=-1), axis=0)
    vort_mse_std = np.std(np.mean(vort_mse, axis=-1), axis=0)

    vel_mse_mean = np.mean(vel_mse_time, axis=-1)
    vort_mse_mean = np.mean(vort_mse_time, axis=-1)

    Y_bars_vel_mse  = [vel_mse_mean[i] for i in range(len(dir))]
    Y_bars_vort_mse = [vort_mse_mean[i] for i in range(len(dir))]

if pargs.reduced:
    reduced_mae_std = np.std(np.mean(reduced_mae, axis=-1), axis=0)
    reduced_mae_mean = np.mean(reduced_mae_time, axis=-1)
    Y_bars_reduced_mae  = [reduced_mae_mean[i] for i in range(len(dir))]

if pargs.mae:

    plt.bar(X_bars, Y_bars_vel_mae, color = colors, alpha = 0.6)
    plt.errorbar(X_bars, Y_bars_vel_mae, yerr = [vel_mae_std[i] for i in range(len(dir))],
    fmt='None', ecolor='black', elinewidth= 0.5, capsize=3, capthick=0.5)
    plt.ylabel('MAE velocity')
    plt.title('Forced turbulence')
    plt.grid(axis='y')
    plt.tight_layout(pad=0.5)
    plt.savefig('forced_turb_velo_mae.pdf')
    plt.close()

    plt.bar(X_bars[1:], Y_bars_vel_improv_mae, color = colors[1:], alpha = 0.6)
    plt.errorbar(X_bars[1:], Y_bars_vel_improv_mae, yerr = [vel_improv_mae_std[i] for i in range(len(dir)-1)],
    fmt='None', ecolor='black', elinewidth= 0.5, capsize=3, capthick=0.5)
    plt.ylabel('Velocity improvement MAE')
    plt.grid(axis='y')
    plt.tight_layout(pad=0.5)
    plt.savefig('forced_turb_velo_improv_mae.pdf')
    plt.close()

    plt.bar(X_bars[1:], Y_bars_vort_improv_mae, color = colors[1:], alpha = 0.6)
    plt.errorbar(X_bars[1:], Y_bars_vort_improv_mae, yerr = [vort_improv_mae_std[i] for i in range(len(dir)-1)],
    fmt='None', ecolor='black', elinewidth= 0.5, capsize=3, capthick=0.5)
    plt.ylabel('Vorticity improvement MAE')
    plt.grid(axis='y')
    plt.tight_layout(pad=0.5)
    plt.savefig('forced_turb_vort_improv_mae.pdf')
    plt.close()


if pargs.reduced:

    plt.bar(X_bars_reduced, Y_bars_reduced_mae, color = colors, alpha = 0.6, hatch='/')
    plt.errorbar(X_bars_reduced, Y_bars_reduced_mae, yerr = [reduced_mae_std[i] for i in range(len(dir))],
    fmt='None', ecolor='black', elinewidth= 0.5, capsize=3, capthick=0.5)
    plt.ylabel('MAE to lerp(ref)')
    plt.title('Forced turbulence')
    plt.grid(axis='y')
    plt.tight_layout(pad=0.5)
    plt.savefig('forced_turb_reduced.pdf')
    plt.close()

print(vel_mae_mean)
print(vel_mse_mean)
print(vort_mae_mean)
print(vort_mse_mean)