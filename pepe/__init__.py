"""

# <u>P</u>hoto<u>e</u>lastic <u>P</u>ython <u>E</u>nvironment

This is a collection of tools for working with photoelastic particle images, including common analysis methods like particle tracking and community analysis.

The vast majority of these methods are either derivative or directly sourced from previous works (see, e.g., [1] for a nice review of techniques); therefore the value of this library, as I see it, is not in novel implementations of methods, but as a single toolbox containing consistent and compatable implementations of a wide selection of tools.

## Features

- Common analysis techniques (G<sup>2</sup>, D<sup>2</sup><sub>min</sub>, etc.)
- Particle position and orientation tracking
- Masking and other preprocessing tools
- Synthetic photoelastic response generation
- Inverse force solving (as in [2])

## Installation

The library is available on PyPi:
```
pip install pepe-granular
```

It can also be installed from the Github repository:

```
git clone https://github.com/Jfeatherstone/pepe
cd pepe
pip install .
```

Theoretically, this should work for any Python >3, but Python 3.7 and 3.11 have been explicitly tested.

Dependencies:

- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [lmfit](https://lmfit.github.io/lmfit-py/index.html)
- [scikit-learn](https://scikit-learn.org/stable/)
- [opencv](https://opencv.org/)
- [Pillow](https://pillow.readthedocs.io/en/stable/)
- [numba](https://numba.pydata.org/)
- [tqdm](https://github.com/tqdm/tqdm)

These can all be installed (alongside their dependencies) via `pip`:
```
git clone https://github.com/Jfeatherstone/pepe
cd pepe
pip install -r requirements.txt
```

Some of the test notebooks may also make use of the Matlab API to compare against Jonathan Kollmer's code, but this is not required to use any functions in the library itself. Installing the Matlab API requires a local installation of Matlab proper; see [here](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) for more information.

If you'd rather not install systemwide (say, to run just a simple example), you can also just manually add the package to your path at the top of a Jupyter notebook/python file with (after cloning the repo as above):

```
import sys
sys.path.append('/absolute/path/to/clone/folder/pepe/')
```

Many of the notebooks in the main directory follow this process, since they are used to test the development versions of the packages.

## Usage

See the `pepe.examples` submodule for some common uses of the toolbox. Many pseudo-unit-tests can be found in the repo's [`notebooks`](https://github.com/Jfeatherstone/pepe/tree/master/notebooks) folder, which may also help determine how to use a certain function. Note that these latter examples are more made for my own benefit, and won't be as heavily commented as the proper examples.

## Further Reading and References

[1] Abed Zadeh, A., Barés, J., Brzinski, T. A., Daniels, K. E., Dijksman, J., Docquier, N., Everitt, H. O., Kollmer, J. E., Lantsoght, O., Wang, D., Workamp, M., Zhao, Y., & Zheng, H. (2019). Enlightening force chains: A review of photoelasticimetry in granular matter. Granular Matter, 21(4), 83. https://doi.org/10.1007/s10035-019-0942-2

[2] Daniels, K. E., Kollmer, J. E., & Puckett, J. G. (2017). Photoelastic force measurements in granular materials. Review of Scientific Instruments, 88(5), 051808. https://doi.org/10.1063/1.4983049

[2a] Jonathan Kollmer's implementation in Matlab: https://github.com/jekollmer/PEGS

[2b] Olivier Lantsoght's implementation in Python: https://git.immc.ucl.ac.be/olantsoght/pegs_py

[3] Photoelastic methods wiki. https://git-xen.lmgc.univ-montp2.fr/PhotoElasticity/Main/-/wikis/home
"""

__version__ = '1.2.7'
__author__ = 'Jack Featherstone'
__credits__ = 'North Carolina State University; Okinawa Institute of Science and Technology' 
