# Non-Gaussian State Simulations

This repository contains simulations of heralded non-Gaussian optical state generation using **MrMustard**.

The main notebook, `cat_state_gen.ipynb`, focuses on:
- generating multi-component cat states,
- visualizing Wigner functions,
- identifying grid-like phase-space structure,
- comparing against finite-energy GKP-inspired references.

<img width="584" height="483" alt="Wigner_with_lattice" src="https://github.com/user-attachments/assets/62ad3ab9-5dcc-4186-a5b3-79acdf13aa62" />


## Quick start

```bash
pip install -r requirements.txt
jupyter lab
```

Open `cat_state_gen.ipynb` and run all cells.

## Environment Notes

This repository now targets the latest **MrMustard** API.
The simulation helpers in `circuits_mrmustard.py` have been migrated accordingly.

Recommended workflow:

```bash
# create a Python 3.11 virtual environment
python3.11 -m venv simulations
source simulations/bin/activate

# install project dependencies
pip install -r requirements.txt
```

## Optional: LaTeX rendering

Some plots use LaTeX text rendering in matplotlib.

If labels do not render correctly, install a LaTeX distribution
