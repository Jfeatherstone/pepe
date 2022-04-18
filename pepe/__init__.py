"""

# <u>P</u>hoto<u>e</u>lastic <u>P</u>ython <u>E</u>nvironment

This is a collection of tools for working with photoelastic particle images, including common analysis methods like particle tracking and community analysis.

The vast majority of these methods are either derivative or directly sourced from previous works (see, e.g., [1] for a nice review of techniques); therefore the value of this library, as I see it, is not in novel implementations of methods, but as a single toolbox containing consistent and compatable implementations of a wide selection of tools.

## Features

- Common analysis techniques (G<sup>2</sup>, D2min, etc.)
- Particle tracking
- Masking and other preprocessing tools
- Synthetic photoelastic response generation
- Force solving (as in [2])

## Installation

The library will (soon) be available on PyPi, meaning the package and dependencies can be installed via:
```
pip install pepe-granular
```

Theoretically, this should work for any Python >3, but Python 3.7 is the recommended version (primarily because OpenCV has some issues with newer versions).

Other than the usual scientific computing packages (`numpy`, `scipy`, `matplotlib`) this library makes use of:

- [lmfit](https://lmfit.github.io/lmfit-py/index.html)
- [scikit-learn](https://scikit-learn.org/stable/)
- [OpenCV](https://opencv.org/)
- [Pillow](https://pillow.readthedocs.io/en/stable/)

These can all be installed (alongside their dependencies) via `pip`:
```
pip intsall python-opencv scikit-learn lmfit Pillow
```

Some of the test notebooks may also make use of the Matlab API to compare against Jonathan Kollmer's implementation, but this is not required to use any functions in the library itself. Installing the Matlab API requires a local installation of Matlab proper; see [here](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) for more information.

If you'd rather build from source, the following sequence should do it on most systems:
```
git clone https://github.com/jfeatherstone/pepe
cd pepe
python setup.py install
```

If you'd rather not install systemwide (say, to run just a simple example), you can also just manually add the package to your path at the top of a Jupyter notebook/python file with (after cloning the repo as above):

```
import sys
sys.path.append('/absolute/path/to/clone/folder/pepe/')
```

## Usage

See the `pepe.examples` submodule for some common uses of the toolbox. Many pseduo-unit-tests can be found in the repo's [`tests`](https://github.com/Jfeatherstone/pepe/tree/master/tests) folder, which may also help determine how to use a certain function. Note that these latter examples are more made for my own benefit, and won't be as heavily commented as the proper examples.

## Further Reading and References

[1] Abed Zadeh, A., Barés, J., Brzinski, T. A., Daniels, K. E., Dijksman, J., Docquier, N., Everitt, H. O., Kollmer, J. E., Lantsoght, O., Wang, D., Workamp, M., Zhao, Y., & Zheng, H. (2019). Enlightening force chains: A review of photoelasticimetry in granular matter. Granular Matter, 21(4), 83. https://doi.org/10.1007/s10035-019-0942-2

[2] Daniels, K. E., Kollmer, J. E., & Puckett, J. G. (2017). Photoelastic force measurements in granular materials. Review of Scientific Instruments, 88(5), 051808. https://doi.org/10.1063/1.4983049

[2a] Jonathan Kollmer's implementation in Matlab: https://github.com/jekollmer/PEGS

[2b] Olivier Lantsoght's implementation in Python: https://git.immc.ucl.ac.be/olantsoght/pegs_py


[3] Photoelastic methods wiki. https://git-xen.lmgc.univ-montp2.fr/PhotoElasticity/Main/-/wikis/home
"""

__version__ = '0.3.0'
__author__ = 'Jack Featherstone'
__credits__ = 'North Carolina State University' 
