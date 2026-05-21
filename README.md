This repository contains data and experiments associated to the paper: Toscano-Duran,V., Gonzalez-Diaz,R., and Gutiérrez-Naranjo, M.A., "Persistent Entropy Transform: A Topological Entropy-Based Signature for Shape Analysis and Signal Characterization".

## Usage

To run the Jupyter notebooks correctly, you need to create a Python virtual environment (This has been developed with Python3.10.11):

```bash
virtualenv -p python env
```

Next, activate the virtual environment (we use the source command because we are working in WSL):

```bash
env\Scripts\activate
```

Finally, install the required libraries:

```bash
pip install numpy pandas matplotlib tqdm gudhi tensorflow
```

If Jupyter Notebook cannot find a kernel, you may need to install ipykernel. Run the following command to resolve this issue:

```bash
pip install ipykernel
```

## Repository structure

- `utils.py`: It contains some necessary functions used in the experiments.

- `pet.ipynb`: The notebook of the experiments.

- `data`: It contains data used in experiments

- `results`: It contains the results obtained from the experiments.

- `figures`: It contains the figures obtained from the experiments

- `FiguresPaper.ipynb`: This contains the code used to generate the illustrative figures that appear in the paper. These figures are saved in the 'figures' folder, alongside other figures that have been commented on and generated in the experimental notebook.


## Citation and Reference

If you want to use our code for your experiments, please cite our paper.

For further information, please contact us at: vtoscano@us.es