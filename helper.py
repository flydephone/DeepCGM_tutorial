"""
helper.py - supporting code for Tutorial_DeepCGM.ipynb.

All boilerplate (loss functions, training loops, prediction, plotting,
pretrained-model loading, and the training-evolution animation) lives here
so the notebook itself stays focused on the *what* and the *why* of each step.

The tutorial repository:  https://github.com/flydephone/DeepCGM_tutorial
The upstream DeepCGM code: https://github.com/WUR-AI/DeepCGM
"""

# ---------------------------------------------------------------------------
# Imports & device
# ---------------------------------------------------------------------------
import os
import time
import glob
import urllib.request

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def _ensure_upstream(git_url="https://github.com/WUR-AI/DeepCGM.git"):
    """Clone the upstream DeepCGM repo (code + data + weights) on first run and chdir in.

    No-op when the upstream code (models_aux/) is already on disk, so it is safe to
    import helper from inside an existing clone or after the first Colab run. This is
    what lets the notebook's setup cell be a two-step "download helper.py, import helper".
    """
    if not os.path.isdir('models_aux'):
        if not os.path.isdir('DeepCGM'):
            assert os.system(f"git clone --depth 1 {git_url}") == 0, \
                "git clone failed - is git installed and do you have internet access?"
        os.chdir('DeepCGM')


_ensure_upstream()

import utils
from models_aux.MyDataset    import MyDataSet
from models_aux.NaiveLSTM    import NaiveLSTM
from models_aux.DeepCGM_fast import DeepCGM
from models_aux.MCLSTM_fast  import MCLSTM

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ---------------------------------------------------------------------------
# Constants (kept identical to train.py:153-159)
# ---------------------------------------------------------------------------
OBS_NAME     = ['DVS', 'PAI', 'WLV', 'WST', 'WSO', 'WAGT', 'WRR14']
OBS_COL_NAME = ['TIME'] + OBS_NAME
OBS_LOC      = [OBS_COL_NAME.index(n) for n in OBS_NAME]
LOSS_WEIGHTS = [1, 1, 5, 2, 2, 1, 2]

# (display name, weight subdir, model class, input_mask flag)
MODEL_DIRS = [
    ("LSTM",          "NaiveLSTM_spa_scratch",     NaiveLSTM, False),
    ("MCLSTM",        "MCLSTM_spa_scratch",        MCLSTM,    False),
    ("DeepCGM",       "DeepCGM_spa_scratch",       DeepCGM,   False),
    ("DeepCGM+IM",    "DeepCGM_spa_IM_scratch",    DeepCGM,   True ),
    ("DeepCGM+CG",    "DeepCGM_spa_CG_scratch",    DeepCGM,   False),
    ("DeepCGM+IM+CG", "DeepCGM_spa_IM_CG_scratch", DeepCGM,   True ),
]

setup_seed = utils.setup_seed   # re-export for convenience


# ---------------------------------------------------------------------------
# Data loading / splitting
# ---------------------------------------------------------------------------
def load_dataset(tra_year='2018'):
    """Return file lists (ory, par, wea_fer, spa, int) and the max/min normaliser tuple.

    NOTE: utils.dataset_loader uses glob.glob, whose ordering follows os.listdir
    and is therefore *platform dependent* (Windows = alphabetical, Linux = arbitrary).
    We sort the lists here so the train/test split is identical on every machine
    (Windows / macOS / Colab Linux), which is essential for reproducible results.
    """
    rea_ory, rea_par, rea_wea_fer, rea_spa, rea_int = utils.dataset_loader(
        data_source=f'format_dataset/real_{tra_year}'
    )
    rea_ory     = np.array(sorted(rea_ory))
    rea_par     = np.array(sorted(rea_par))
    rea_wea_fer = np.array(sorted(rea_wea_fer))
    rea_spa     = np.array(sorted(rea_spa))
    rea_int     = np.array(sorted(rea_int))
    max_min = utils.pickle_load('format_dataset/max_min.pickle')
    return rea_ory, rea_par, rea_wea_fer, rea_spa, rea_int, max_min


def split_and_load(rea_ory, rea_wea_fer, rea_spa, rea_int,
                   tra_year='2018', batch_size=128):
    """Train/test split exactly as in the paper (2018 uses first 65 samples as train)."""
    sample_2018 = 65
    if tra_year == "2018":
        tra_idx, tes_idx = slice(0, sample_2018), slice(sample_2018, None)
    else:
        tes_idx, tra_idx = slice(0, sample_2018), slice(sample_2018, None)
    tra_set = MyDataSet(obs_loc=OBS_LOC,
                        ory=rea_ory[tra_idx], wea_fer=rea_wea_fer[tra_idx],
                        spa=rea_spa[tra_idx], int_=rea_int[tra_idx],
                        batch_size=batch_size)
    tes_set = MyDataSet(obs_loc=OBS_LOC,
                        ory=rea_ory[tes_idx], wea_fer=rea_wea_fer[tes_idx],
                        spa=rea_spa[tes_idx], int_=rea_int[tes_idx],
                        batch_size=batch_size)
    return (DataLoader(tra_set, batch_size=batch_size, shuffle=False),
            DataLoader(tes_set, batch_size=batch_size, shuffle=False),
            len(tra_set), len(tes_set))


def show_sample_overview(sample_idx, rea_ory, rea_spa, rea_wea_fer, max_min):
    """Plot one growing season's weather + sparse observations side by side."""
    res_max, res_min, _, _, wea_fer_max, wea_fer_min = max_min
    sample_ory     = utils.pickle_load(rea_ory[sample_idx])
    sample_obs     = utils.pickle_load(rea_spa[sample_idx])
    sample_wea_fer = utils.pickle_load(rea_wea_fer[sample_idx])

    # wea_fer length = the actual growing-season length (~150-180 days);
    # ory / obs are padded to 200 days, so we slice to match.
    L = sample_wea_fer.shape[0]
    day  = sample_wea_fer[:, 0] * (wea_fer_max[0] - wea_fer_min[0]) + wea_fer_min[0]
    rad  = sample_wea_fer[:, 1] * (wea_fer_max[1] - wea_fer_min[1])
    tmin = sample_wea_fer[:, 2] * (wea_fer_max[2] - wea_fer_min[2])
    tmax = sample_wea_fer[:, 3] * (wea_fer_max[3] - wea_fer_min[3])
    ory_un = sample_ory[:L] * (res_max[:8] - res_min[:8]) + res_min[:8]
    obs_un = sample_obs[:L] * (res_max[:8] - res_min[:8]) + res_min[:8]
    fer    = sample_wea_fer[:, 7] * (wea_fer_max[7] - wea_fer_min[7])

    fig, axes = plt.subplots(2, 4, figsize=(14, 5), dpi=100)
    axes[0, 0].plot(day, rad, color='orange')
    axes[0, 0].set_title("Radiation"); axes[0, 0].set_ylabel("kJ/m²/d")
    axes[0, 1].plot(day, tmax, color='red',  label='Tmax')
    axes[0, 1].plot(day, tmin, color='blue', alpha=0.7, label='Tmin')
    axes[0, 1].set_title("Tmax / Tmin"); axes[0, 1].set_ylabel("°C"); axes[0, 1].legend(fontsize=7)
    axes[0, 2].bar(day, fer, width=2, color='green')
    axes[0, 2].set_title(f"N fertilization ({int(fer.sum())} kg/ha total, {(fer>0).sum()} events)")
    axes[0, 2].set_ylabel("kg/ha/day")
    axes[0, 3].plot(day, ory_un[:, 1])
    axes[0, 3].set_title("DVS (ORYZA)")

    for k, name in enumerate(['PAI', 'WLV', 'WST', 'WSO']):
        ax = axes[1, k]; j = OBS_COL_NAME.index(name)
        ax.plot(day, ory_un[:, j], color='gray', label='ORYZA2000')
        valid = ~np.isnan(obs_un[:, j])
        ax.scatter(day[valid], obs_un[valid, j], color='black', s=18, label='observation')
        ax.set_title(name); ax.set_xlabel("DOY")
    for ax in axes.flat: ax.tick_params(labelsize=8)
    axes[1, 0].legend(fontsize=8)
    plt.suptitle(f"Sample #{sample_idx} - one growing season: weather + observations", y=1.02)
    plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------------
# Losses and one training/eval step (verbatim from train.py)
# ---------------------------------------------------------------------------
def FITTING_LOSS(pred, real):
    """Per-variable weighted MSE on positions where observations exist."""
    mse  = nn.MSELoss(reduction='none')(pred, real)
    mask = real.ne(-10000)
    return sum(mse[:, :, i].masked_select(mask[:, :, i]).mean() * w
               for i, w in enumerate(LOSS_WEIGHTS))


def CG_LOSS(mask, aux_all):
    """Penalty enforcing smooth transitions of the internal cell state."""
    C_cell_all, C_cell_conv_all, num_segment = aux_all
    mse    = nn.MSELoss(reduction='none')(C_cell_conv_all, C_cell_all)
    mask_p = mask[:, :-1, 0]
    return mse.masked_select(mask_p[:, :, None].repeat(1, 1, mse.shape[-1])).mean() * 1e5


def run_one_epoch(model, loader, mode, optimizer=None,
                  convergence_loss=False, target='spa'):
    """One pass over `loader`. Returns mean (sparse, interpolated, CG) losses."""
    if mode == 'tra':
        model.train()
    else:
        model.eval()
    L_spa, L_int, L_cg = 0., 0., 0.
    ctx = torch.enable_grad() if mode == 'tra' else torch.no_grad()
    with ctx:
        for x, ory, spa, int_ in loader:
            x, ory, spa, int_ = x.to(device), ory.to(device), spa.to(device), int_.to(device)
            mask     = spa.ne(-10000)
            # model input = wea_fer columns [1,2,3,7,8] = [Rad, Tmax, Tmin, N_fert, aux]
            pred, aux = model(x[:, :, [1, 2, 3, 7, 8]], ory)
            loss_spa = FITTING_LOSS(pred, spa)
            loss_int = FITTING_LOSS(pred, int_)
            loss_cg  = CG_LOSS(mask, aux) if len(aux) > 0 else torch.tensor(0.)
            if mode == 'tra':
                optimizer.zero_grad()
                base  = loss_spa if target == 'spa' else loss_int
                total = base + loss_cg * 1.0 * float(convergence_loss) if len(aux) > 0 else base
                total.backward()
                optimizer.step()
            L_spa += float(loss_spa); L_int += float(loss_int); L_cg += float(loss_cg)
    n = len(loader)
    return L_spa / n, L_int / n, L_cg / n


# ---------------------------------------------------------------------------
# Prediction / plotting / RMSE
# ---------------------------------------------------------------------------
def predict(model, loader, max_min):
    """Run the model over `loader` and return un-normalised numpy arrays."""
    res_max, res_min, _, _, wea_fer_max, wea_fer_min = max_min
    model.eval()
    pres, spas, orys, ints, was = [], [], [], [], []
    with torch.no_grad():
        for x, ory, spa, int_ in loader:
            x, ory = x.to(device), ory.to(device)
            pred, _ = model(x[:, :, [1, 2, 3, 7, 8]], ory)
            unsca = lambda t, mx, mn: utils.unscalling(t.detach().cpu().numpy(), mx, mn)
            pres.append(unsca(pred,  res_max[OBS_LOC], res_min[OBS_LOC]))
            orys.append(unsca(ory,   res_max[OBS_LOC], res_min[OBS_LOC]))
            spas.append(unsca(spa,   res_max[OBS_LOC], res_min[OBS_LOC]))
            ints.append(unsca(int_,  res_max[OBS_LOC], res_min[OBS_LOC]))
            was.append(unsca(x, wea_fer_max, wea_fer_min))
    return [np.concatenate(arr, 0) for arr in (pres, spas, orys, ints, was)]


def plot_one_sample(pre, spa, ory, wea, sample_loc=-1, title=""):
    """Compact six-panel plot (PAI / WLV / WST / WSO / WAGT / Yield) for one sample."""
    fig, axs = plt.subplots(1, 6, figsize=(15, 2.6), dpi=100)
    day = wea[sample_loc, :, 0]
    for k, name in enumerate(['PAI', 'WLV', 'WST', 'WSO', 'WAGT', 'WRR14']):
        ax = axs[k]; j = OBS_NAME.index(name)
        ory_v, obs_v, pre_v = ory[sample_loc, :, j], spa[sample_loc, :, j], pre[sample_loc, :, j]
        m_ory = (day >= 0) & (ory_v >= 0)   # filter padding AND ORY-invalid days
        m_obs = (day >= 0) & (obs_v >= 0)
        ax.plot(day[m_ory],    ory_v[m_ory], color='gray', label='ORYZA2000')
        ax.scatter(day[m_obs], obs_v[m_obs], s=12, color='black', label='obs')
        ax.plot(day[m_ory],    pre_v[m_ory], color='red', linewidth=1.2, label='model')
        ax.set_title(name.replace('WRR14', 'Yield')); ax.set_xlabel('DOY')
    axs[0].legend(fontsize=7, loc='upper left')
    plt.suptitle(title, y=1.05); plt.tight_layout(); plt.show()


def rmse_per_var(pre, spa):
    """Per-variable RMSE on positions where sparse observations exist."""
    out = {}
    for i, name in enumerate(OBS_NAME):
        m = spa[:, :, i] >= 0
        out[name] = float(np.sqrt(((pre[:, :, i][m] - spa[:, :, i][m])**2).mean())) if m.sum() else np.nan
    return out


def show_task_result(tag, pretrained_predictions, title=None, sample_loc=-1):
    """Render the standard 'one sample plot + per-variable RMSE' block for a task."""
    pre, spa, ory, int_, wea = pretrained_predictions[tag]
    plot_one_sample(pre, spa, ory, wea, sample_loc=sample_loc,
                    title=title or f"Task: {tag}")
    print("RMSE for each variable (lower is better):")
    for k, v in rmse_per_var(pre, spa).items():
        print(f"  {k:6s} : {v:10.3f}")


# ---------------------------------------------------------------------------
# Pretrained model loading
# ---------------------------------------------------------------------------
def load_best_pretrained(model_dir, model_cls, input_mask, tra_year='2018', seed=0):
    """Pick the SEED-th robust run for this config and return its model + filename.

    NOTE: os.listdir() ordering is platform dependent; we sort to ensure that
    `runs[seed]` picks the same robust run on every machine (Windows / Colab).
    The run sub-directories are named like `2018_2018_<seed>_<timestamp>`, so
    sorting alphabetically also sorts by the embedded seed number.
    """
    runs = sorted(r for r in os.listdir(f'model_weight/{model_dir}/') if tra_year in r)
    run  = runs[seed]
    path = f'model_weight/{model_dir}/{run}'
    files = os.listdir(path)
    tra_losses = [float(f[:-4].split('_')[-3]) for f in files]   # ..._tra_0.0241_tes_0.0393.pkl
    best = files[int(np.argmin(tra_losses))]
    if model_cls is NaiveLSTM:
        m = NaiveLSTM().to(device)
    else:
        m = model_cls(input_mask=input_mask).to(device)
    m.load_state_dict(torch.load(os.path.join(path, best), map_location=device), strict=True)
    return m, best


def load_all_pretrained(tes_loader, max_min, tra_year='2018', seed=0):
    """Load every configuration in MODEL_DIRS and cache its test-set predictions."""
    cached = {}
    for tag, dir_, cls, im in tqdm(MODEL_DIRS,
                                    desc="Loading 6 pretrained 700-epoch models",
                                    unit="model"):
        m, _ = load_best_pretrained(dir_, cls, im, tra_year, seed)
        cached[tag] = predict(m, tes_loader, max_min)
    print("All six 700-epoch pretrained models loaded and cached.")
    return cached


# ---------------------------------------------------------------------------
# Comprehensive comparison (Part 6 of the notebook)
# ---------------------------------------------------------------------------
def compare_models(pretrained_predictions, sample_loc=-1):
    """6-row x 6-column grid: rows = variables, columns = configurations."""
    var_to_plot = ['PAI', 'WLV', 'WST', 'WSO', 'WAGT', 'WRR14']
    max_vals    = [8, 6000, 6000, 8000, 14000, 10000]
    fig, axs = plt.subplots(len(var_to_plot), len(MODEL_DIRS),
                            figsize=(13, 9), dpi=100)
    for i, vname in enumerate(var_to_plot):
        j_var = OBS_NAME.index(vname)
        for j, (tag, *_) in enumerate(MODEL_DIRS):
            pre, spa, ory, int_, wea = pretrained_predictions[tag]
            ax = axs[i, j]
            day   = wea[sample_loc, :, 0]
            ory_v = ory[sample_loc, :, j_var]
            obs_v = spa[sample_loc, :, j_var]
            pre_v = pre[sample_loc, :, j_var]
            m_ory = (day >= 0) & (ory_v >= 0)
            m_obs = (day >= 0) & (obs_v >= 0)
            ax.plot(day[m_ory],    ory_v[m_ory], color='gray', lw=1)
            ax.scatter(day[m_obs], obs_v[m_obs], s=8, color='black')
            ax.plot(day[m_ory],    pre_v[m_ory], color='red', lw=1)
            ax.set_ylim(bottom=0, top=max_vals[i])
            if i == 0: ax.set_title(tag, fontsize=9)
            if j == 0: ax.set_ylabel(vname.replace('WRR14', 'Yield'), fontsize=9)
            ax.tick_params(labelsize=7)
    plt.suptitle("Model comparison on the same test sample (700-epoch pretrained)", y=1.01)
    plt.tight_layout(); plt.show()


def rmse_table(pretrained_predictions):
    """Pandas DataFrame of per-variable RMSE for every configuration."""
    rows = []
    for tag, *_ in MODEL_DIRS:
        pre, spa, *_ = pretrained_predictions[tag]
        rows.append({'model': tag, **rmse_per_var(pre, spa)})
    df = pd.DataFrame(rows).set_index('model').round(3)
    print("=== Test-set RMSE (lower is better) ===")
    print(df)
    return df


# ---------------------------------------------------------------------------
# Training utilities (used by the bonus training-evolution section)
# ---------------------------------------------------------------------------
def train_loop(model, epochs, lr, tra_loader, tes_loader,
               convergence_loss=False, target='spa', tag=""):
    """Generic training loop. Returns a (epoch, tra_spa, tra_int, tra_cg, tes_spa, ...) log array."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log = []; t0 = time.time()
    pbar = tqdm(range(epochs), desc=f"[{tag}]", unit="ep", leave=True)
    for e in pbar:
        tra = run_one_epoch(model, tra_loader, 'tra', optimizer, convergence_loss, target)
        tes = run_one_epoch(model, tes_loader, 'tes')
        log.append((e, *tra, *tes))
        pbar.set_postfix(tra_spa=f"{tra[0]:.4f}",
                         tes_spa=f"{tes[0]:.4f}",
                         cg=f"{tra[2]:.4f}")
    print(f"  [{tag}] {epochs} epochs done in {time.time()-t0:.1f}s, final tes_spa={tes[0]:.4f}")
    return np.array(log)


def hands_on_training(tra_loader, tes_loader, epochs=20, lr=0.1, seed=0):
    """Run a short DeepCGM+IM+CG training so students can *feel* how slow training is.

    Prints the configuration up front, shows a tqdm bar with ETA during training,
    and then prints a summary with an extrapolation to the paper's full setup
    (700 epochs x 50 robust runs x 6 configurations).  Returns (model, log).
    """
    utils.setup_seed(seed)
    model = DeepCGM(input_mask=True).to(device)

    print(f"Configuration : DeepCGM+IM+CG (input_mask=True, convergence_loss=True)")
    print(f"Device        : {device}  ({'GPU' if str(device).startswith('cuda') else 'CPU'})")
    print(f"Epochs        : {epochs}")
    print(f"Learning rate : {lr}")
    print("Watch the tqdm bar below - the ETA tells you how long the wait is going to be ...")
    print()

    t_start = time.time()
    log = train_loop(model, epochs=epochs, lr=lr,
                     tra_loader=tra_loader, tes_loader=tes_loader,
                     convergence_loss=True, target='spa', tag='hands-on')
    t_elapsed = time.time() - t_start
    per_epoch = t_elapsed / epochs

    print()
    print(f"=== Training summary ===")
    print(f"Wall time for {epochs} epochs : {t_elapsed:>7.1f} s")
    print(f"Per epoch                  : {per_epoch*1000:>7.0f} ms")
    print()
    print(f"Extrapolated cost of the paper's full setup:")
    print(f"  700 epochs                  : {per_epoch * 700:>7.0f} s  (~{per_epoch * 700 / 60:.1f} min)")
    print(f"  700 epochs x 50 robust runs : {per_epoch * 700 * 50 / 3600:>7.1f} hours")
    print(f"  ...x 6 configurations       : {per_epoch * 700 * 50 * 6 / 3600:>7.1f} hours")
    print()
    print("This is why Tasks 1-2.2 above load pretrained weights instead of training from scratch,")
    print("and why section 6.2 below shows a pre-rendered GIF rather than re-running 700 epochs here.")
    return model, log


def train_with_snapshots(model_cls, input_mask, lr, convergence_loss, tag,
                         total_epochs, snap_every, tra_loader, tes_loader, max_min, seed=0):
    """Train and store test-set predictions every `snap_every` epochs (used for the GIF)."""
    utils.setup_seed(seed)
    if model_cls is NaiveLSTM:
        m = NaiveLSTM().to(device)
    else:
        m = model_cls(input_mask=input_mask).to(device)
    optimizer = torch.optim.Adam(m.parameters(), lr=lr)
    snapshots = []
    pbar = tqdm(range(total_epochs), desc=f"[{tag}]", unit="ep")
    for e in pbar:
        if e % snap_every == 0 or e == total_epochs - 1:
            pre, spa, ory, _, wea = predict(m, tes_loader, max_min)
            snapshots.append((e, pre, spa, ory, wea))
        tra = run_one_epoch(m, tra_loader, 'tra', optimizer, convergence_loss, target='spa')
        pbar.set_postfix(tra_spa=f"{tra[0]:.4f}")
    pre, spa, ory, _, wea = predict(m, tes_loader, max_min)
    snapshots.append((total_epochs, pre, spa, ory, wea))
    return snapshots


def make_evolution_gif(lstm_snaps, dcgm_snaps, gif_path='figure/training_evolution.gif',
                       fps=5, sample_loc=-1, total_epochs=None):
    """Build a 2-row (LSTM vs DeepCGM+IM+CG) GIF from snapshot lists. Returns frame count."""
    var_names = ['PAI', 'WLV', 'WST', 'WSO', 'WAGT', 'WRR14']
    max_vals  = [8, 6000, 6000, 8000, 14000, 10000]
    N_FRAMES  = min(len(lstm_snaps), len(dcgm_snaps))
    total_epochs = total_epochs or lstm_snaps[-1][0]
    fig, axs = plt.subplots(2, 6, figsize=(15, 5), dpi=80)

    def draw_frame(idx):
        for r in range(2):
            for c in range(6):
                axs[r, c].clear()
        rows = [("LSTM",          lstm_snaps, 'red' ),
                ("DeepCGM+IM+CG", dcgm_snaps, 'blue')]
        for row, (label, snaps, color) in enumerate(rows):
            ep, pre, spa, ory, wea = snaps[idx]
            day = wea[sample_loc, :, 0]
            for col, vname in enumerate(var_names):
                ax = axs[row, col]
                j = OBS_NAME.index(vname)
                ory_v = ory[sample_loc, :, j]
                obs_v = spa[sample_loc, :, j]
                pre_v = pre[sample_loc, :, j]
                m_ory = (day >= 0) & (ory_v >= 0)
                m_obs = (day >= 0) & (obs_v >= 0)
                ax.plot(day[m_ory],    ory_v[m_ory], color='gray',  lw=1)
                ax.scatter(day[m_obs], obs_v[m_obs], s=12, color='black')
                ax.plot(day[m_ory],    pre_v[m_ory], color=color,  lw=1.6)
                ax.set_ylim(bottom=0, top=max_vals[col])
                ax.tick_params(labelsize=7)
                if row == 0:
                    ax.set_title(vname.replace('WRR14', 'Yield'), fontsize=10)
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel('DOY', fontsize=8)
                if col == 0:
                    ax.set_ylabel(label, fontsize=10, color=color, fontweight='bold')
        fig.suptitle(f"Training evolution - epoch {ep:3d} / {total_epochs}",
                     y=0.99, fontsize=13)

    ani = animation.FuncAnimation(fig, draw_frame, frames=N_FRAMES,
                                  interval=1000 / fps, repeat=True)
    utils.find_or_make(os.path.dirname(gif_path) or '.')
    ani.save(gif_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return N_FRAMES


def show_evolution_gif(gif_path='figure/training_evolution.gif',
                       url='https://raw.githubusercontent.com/flydephone/DeepCGM_tutorial/main/figure/training_evolution.gif'):
    """Download the pre-rendered 700-epoch training-evolution GIF if missing, then
    return it as an IPython Image so the notebook can display it inline with a one-liner."""
    utils.find_or_make('figure')
    if not os.path.exists(gif_path):
        print("Downloading pre-rendered 700-epoch GIF ...")
        urllib.request.urlretrieve(url, gif_path)
    print(f"GIF on disk: {gif_path}  ({os.path.getsize(gif_path)/1024:.0f} KB)")
    from IPython.display import Image
    return Image(gif_path)
