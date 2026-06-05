# DeepCGM Tutorial

Hands-on Jupyter notebook tutorial for the master-level course on **data-driven crop
modelling**, built around the **DeepCGM** paper:

> Han et al. (2025). *Knowledge-guided machine learning with multivariate sparse data for
> crop growth modelling.* Field Crops Research.
> [DOI 10.1016/j.fcr.2025.109885](https://doi.org/10.1016/j.fcr.2025.109885)

Upstream code & paper resources: <https://github.com/WUR-AI/DeepCGM>

The tutorial walks students step by step through the four experiments on slide 6 of the
course slides:

1. **Task 1** — fit sparse observations with a NaiveLSTM (data-driven baseline).
2. **Task 2.1** — fit with DeepCGM using mass-conservation only.
3. **Task 2.2** — add the Input Mask (relevant-inputs-only constraint).
4. **Task 2.3** — add the Convergence Loss (stable internal processes).

All required tasks load the **pretrained 700-epoch weights** released by the original
authors, so students do **not** have to train anything themselves. A bonus section at
the end retrains LSTM and DeepCGM+IM+CG from scratch and renders an animated GIF
of the training evolution.

---

## Quick start

```bash
git clone https://github.com/flydephone/DeepCGM_tutorial.git
cd DeepCGM_tutorial
conda create -n DeepCGM python==3.10.16
conda activate DeepCGM
pip install -r requirements.txt
jupyter lab Tutorial_DeepCGM.ipynb
```

Then open `Tutorial_DeepCGM.ipynb` and run the cells in order. The notebook already
contains baked outputs, so you can also just read it without executing anything.

CPU is sufficient for everything in the required sections. The bonus *Training evolution*
section retrains two models for 700 epochs and takes **~15 minutes on CPU**.

---

## Repository layout

```
DeepCGM_tutorial/
├── Tutorial_DeepCGM.ipynb         The main notebook (with outputs already baked in)
├── utils.py                       Data loading, normalisation, plotting helpers
├── train.py                       Original training script (referenced in the notebook)
├── requirements.txt               numpy / pandas / matplotlib / scipy / torch / tqdm
├── models_aux/
│   ├── NaiveLSTM.py               Plain LSTM baseline
│   ├── DeepCGM.py                 DeepCGM reference implementation (explicit gates)
│   ├── DeepCGM_fast.py            Optimised DeepCGM used at training time
│   ├── MCLSTM.py / MCLSTM_fast.py Mass-conserving LSTM (Hoedt et al., 2021)
│   └── MyDataset.py               PyTorch Dataset wrapper for the formatted data
├── format_dataset/                Pre-formatted rice field data (2018 + 2019 seasons)
├── model_weight/                  Released 700-epoch checkpoints for 6 configurations
│   ├── NaiveLSTM_spa_scratch
│   ├── MCLSTM_spa_scratch
│   ├── DeepCGM_spa_scratch
│   ├── DeepCGM_spa_IM_scratch
│   ├── DeepCGM_spa_CG_scratch
│   └── DeepCGM_spa_IM_CG_scratch
└── figure/
    └── training_evolution.gif     GIF produced by the bonus section
```

The data and model weights are an exact subset of the upstream DeepCGM release
(only the six configurations needed by the tutorial are included to keep the repo
small).

---

## License

This tutorial inherits the **CC BY-NC 4.0** license of the upstream DeepCGM
repository: free for non-commercial research and academic use.

For commercial enquiries, contact the original author at `hanjingye@whu.edu.cn`.

See [LICENSE.md](LICENSE.md) for the full text.
