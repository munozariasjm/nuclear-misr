# Nuclear PyModels

Python implementation of nuclear models obtained in the paper [Discovering Nuclear Models from Symbolic Machine Learning](https://arxiv.org/abs/2404.11477).

Interact with the models online using the [nuclear-pymodels web app](https://nuclear-misr.streamlit.app/).

## Installation

First clone the repository

```bash
git clone https://github.com/munozariasjm/nuclear-misr.git
```

Then install the package

```bash
cd nuclear-misr
pip install .
```

## Usage

```python
from nuclearpy_models.models.BE import sr_be
sr_be(56, 26)
```

## Reproducing the results

You can find the notebooks to test and generate the plots for the paper in:
- Charge radii: [notebooks/charge_radii.ipynb](https://github.com/munozariasjm/nuclear-misr/blob/master/notebooks/test/rc.ipynb)
- Energy: [notebooks/test/be](https://github.com/munozariasjm/nuclear-misr/blob/master/notebooks/test/be.ipynb)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Authors
- [Jose M Munoz](https://munozariasjm.github.io/)
- [Silviu M Udrescu](https://scholar.google.com/citations?user=maphp-0AAAAJ&hl=en)
- [Ronald F Garcia Ruiz](https://physics.mit.edu/faculty/ronald-garcia-ruiz/)

[Laboratory of Exotic Molecules and Atoms](https://www.garciaruizlab.com/)
Laboratory for Nuclear Science, Massachusetts Institute of Technology.