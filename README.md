# **P**hoto**e**lastic **P**ython **E**nvironment

This is a collection of tools for working with photoelastic particle images, including common analysis methods like particle tracking and community analysis.

## Features

- Common analysis techniques (G<sup>2</sup>, D2min, etc.)
- Particle tracking
- Masking and other preprocessing tools
- Synthetic photoelastic response generation
- Force solving (a la PeGS)

## Documentation

Maybe one day :)

## Usage

Wiki is WIP

## Notes

(To be moved to the wiki eventually)

### Force Solving

This library's implementation of force solution differs from PeGS in a few ways that it is important to understand if you are switching between the two:

- In most physical scenarios, it is most common to write coordinates, as [x,y], whereas in image processing it is more common to use [y,x]. This library uses the latter [y,x] convention pretty much everywhere.
- Almost all positions/lengths/etc. are kept in units of pixels for most operations. Some methods may require physical units, for which you may need to pass a parameter along the lines of `pxPerMeter` -- but you should never convert to mm/cm/m manually unless explicitly stated.
- This version does not center around a `particle` struct that contains all of the information about any given particle. The various quantities (active forces, positions, angles, etc.) are left to the user's organization scheme. My reasoning here is that this is much more extensible, since we are not tied to a particular system to run every analysis -- ie. if we want to quickly test some method, we do not need to spend any time organizing every possible parameter a particle could have.

## Further Reading and References

[1] Daniels, K. E., Kollmer, J. E., & Puckett, J. G. (2017). Photoelastic force measurements in granular materials. Review of Scientific Instruments, 88(5), 051808. https://doi.org/10.1063/1.4983049

[1a] Jonathan Kollmer's implementation in Matlab: https://github.com/jekollmer/PEGS

[1b] Olivier Lantsoght's implementation in Python: https://git.immc.ucl.ac.be/olantsoght/pegs_py

[2] Abed Zadeh, A., Barés, J., Brzinski, T. A., Daniels, K. E., Dijksman, J., Docquier, N., Everitt, H. O., Kollmer, J. E., Lantsoght, O., Wang, D., Workamp, M., Zhao, Y., & Zheng, H. (2019). Enlightening force chains: A review of photoelasticimetry in granular matter. Granular Matter, 21(4), 83. https://doi.org/10.1007/s10035-019-0942-2


