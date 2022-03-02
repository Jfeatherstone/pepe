# <ins>P</ins>hoto<ins>e</ins>lastic <ins>P</ins>ython <ins>E</ins>nvironment

This is a collection of tools for working with photoelastic particle images, including common analysis methods like particle tracking and community analysis.

## Features

- Common analysis techniques (G<sup>2</sup>, D<sup>2</sup>min, etc.)
- Particle tracking
- Masking and other preprocessing tools
- Synthetic photoelastic response generation
- Force solving (a la PeGS [1a, 1b])

## Installation

The package is available by cloning from the git repo:

```
git clone https://github.com/Jfeatherstone/pepe
cd pepe
pip install .
```

## Documentation

Available [here](http://jfeatherstone.github.io/pepe/pepe).

## Requirements

Python 3.7 is the recommended version to use, with the following packages:

- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [lmfit](https://lmfit.github.io/lmfit-py/index.html)
- [scikit-learn](https://scikit-learn.org/stable/)
- [OpenCV](https://opencv.org/)
- [Pillow](https://pillow.readthedocs.io/en/stable/)
- [Numba](https://numba.pydata.org/)

These can all be installed (alongside their dependencies) via `pip`:
```
pip install numpy matplotlib python-opencv scikit-learn lmfit Pillow numba
```

Some of the test notebooks may also make use of the Matlab API to compare against Jonathan Kollmer's implementation [1a], but this is not required to use any functions in the library itself. Installing the Matlab API requires a local installation of Matlab proper; see [here](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) for more information.

## Usage

The [wiki](https://github.com/Jfeatherstone/pepe/wiki) and [documentation](http://jfeatherstone.github.io/pepe/pepe) contain information about how to use the toolbox. All test notebooks can be
found in the repo's [`tests`](https://github.com/Jfeatherstone/pepe/tree/master/tests) directory.

## Further Reading and References

[1] Daniels, K. E., Kollmer, J. E., & Puckett, J. G. (2017). Photoelastic force measurements in granular materials. Review of Scientific Instruments, 88(5), 051808. https://doi.org/10.1063/1.4983049

[1a] Jonathan Kollmer's implementation in Matlab: https://github.com/jekollmer/PEGS

[1b] Olivier Lantsoght's implementation in Python: https://git.immc.ucl.ac.be/olantsoght/pegs_py

[2] Abed Zadeh, A., Barés, J., Brzinski, T. A., Daniels, K. E., Dijksman, J., Docquier, N., Everitt, H. O., Kollmer, J. E., Lantsoght, O., Wang, D., Workamp, M., Zhao, Y., & Zheng, H. (2019). Enlightening force chains: A review of photoelasticimetry in granular matter. Granular Matter, 21(4), 83. https://doi.org/10.1007/s10035-019-0942-2

[3] Photoelastic methods wiki. https://git-xen.lmgc.univ-montp2.fr/PhotoElasticity/Main/-/wikis/home
