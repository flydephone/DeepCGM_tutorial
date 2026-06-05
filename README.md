# DeepCGM Tutorial

Hands-on Jupyter notebook tutorial for the master-level course on **data-driven
crop modelling**, built around the **DeepCGM** paper:

> Han et al. (2025). *Knowledge-guided machine learning with multivariate sparse
> data for crop growth modelling.* Field Crops Research.
> [DOI 10.1016/j.fcr.2025.109885](https://doi.org/10.1016/j.fcr.2025.109885)

Upstream code & paper resources: <https://github.com/WUR-AI/DeepCGM>

The tutorial walks students step by step through the four experiments on
slide 6 of the course slides:

1. **Task 1** - fit sparse observations with a NaiveLSTM (data-driven baseline).
2. **Task 2.1** - fit with DeepCGM using mass-conservation only.
3. **Task 2.2** - add the Input Mask (relevant-inputs-only constraint).
4. **Task 2.3** - add the Convergence Loss (stable internal processes).

All four required tasks rely on the **700-epoch pretrained weights** released by
the original authors, so students do **not** have to train anything themselves.
A bonus section at the end retrains LSTM and DeepCGM+IM+CG from scratch and
animates the training evolution as a GIF.

## How to run

### Option A: Google Colab (zero-install)

1. Open Colab.
2. **File -> Open notebook -> GitHub -> URL**:
   `https://github.com/flydephone/DeepCGM_tutorial/blob/main/Tutorial_DeepCGM.ipynb`
3. **Runtime -> Run all**.

The first cell automatically clones the upstream DeepCGM repository and
downloads `helper.py` from this tutorial repo, so nothing else needs to be
installed.

### Option B: your own machine

```bash
# Pick or create an empty working directory
mkdir DeepCGM_tutorial_workspace && cd DeepCGM_tutorial_workspace

# Grab just the two tutorial files
curl -O https://raw.githubusercontent.com/flydephone/DeepCGM_tutorial/main/Tutorial_DeepCGM.ipynb
curl -O https://raw.githubusercontent.com/flydephone/DeepCGM_tutorial/main/helper.py

# Set up an environment (matches the upstream requirements)
conda create -n DeepCGM python==3.10.16
conda activate DeepCGM
pip install numpy==1.26.4 pandas matplotlib scipy tqdm jupyterlab
pip install torch==2.2.2

jupyter lab Tutorial_DeepCGM.ipynb
```

Run the cells in order. The first cell will clone the upstream DeepCGM
repository into a `DeepCGM/` subdirectory and continue from there.

> The required sections need CPU only. The bonus *training evolution* section
> retrains two models for 700 epochs and takes ~15 minutes on CPU.

## What lives in this repository

```
DeepCGM_tutorial/
├── README.md                      You are here
├── LICENSE.md                     CC BY-NC 4.0 (inherited from upstream)
├── Tutorial_DeepCGM.ipynb         The main notebook (slim, focused on narrative)
└── helper.py                      All boilerplate: losses, training loop,
                                    plotting, pretrained loading, GIF rendering
```

Everything else (model definitions, formatted data, pretrained weights) is
pulled at run-time from the upstream
[`WUR-AI/DeepCGM`](https://github.com/WUR-AI/DeepCGM) repository - that way this
tutorial repo stays under a megabyte.

## License

CC BY-NC 4.0 - free for non-commercial research and academic use. Commercial
enquiries should be sent to `hanjingye@whu.edu.cn`.
