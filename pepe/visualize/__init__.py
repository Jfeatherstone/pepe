"""

Methods for visualizing the results of other submodules, like displaying particle
positions, forces, or contact angles.

Listed below are some important tips and conventions that are
used (both within this submodule and externally) that will be
helpful to keep in mind as you are working with this library.

Visualization Tips
------------------

### Coordinate Order

Nearly every part of this library will return coordinates as `[y,x]`,
which is the reverse of the typical order `[x,y]`. This is done because
the majority of image processing techniques/analysis methods utilize the
`[y,x]` ordering, and so we stick to it as well. You will also be expected to
input most quantities in this order eg. the center coordinates for a circular
mask created with `pepe.preprocess.circularMask()`.

Matplotlib, on the other hand, normally assumes `[x,y]` ordering, and so it
may be necessary to flip the order upon graphing any positions. This is very
easy to do with array slicing:

```
a = np.array([[1, 2], [3, 4], [5, 6]])

# Normal order
print(a) # Output: [[1, 2], [3, 4], [5, 6]]

# Flipped order
print(a[:,::-1]) # Output: [[2, 1], [4, 3], [6, 5]]
```

### Flipped Y-Axis

Another convention of images is to invert the y-axis when displaying an
image, such that `[0,0]` is in the top-left corner of the plot. This is
the case whenever you display something with `plt.imshow()`, but will not
be implemented if you simply plot circles with `plt.Circle()` (or similar).

As a result, it may be necessary to flip the axis manually with:

```
ax.invert_yaxis()
```

If this is not done, the positions of contact points may look inverted, which
can be especially hard to notice if you system has vertical symmetry (since
it won't be obvious by just the positions of the particles).
"""

from .Colors import *
from .Circles import *
from .Forces import *
from .Rotation import *
