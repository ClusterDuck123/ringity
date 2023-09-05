import warnings
import numpy as np

from itertools import compress
from collections.abc import MutableMapping
from ringity.utils.plotting import plot_dgm
from ringity.ringscore.metric2ringscore import ring_score_from_sequence
from ringity.utils.exceptions import (
    SchroedingersException,
    TimeParadoxError,
    EndOfTimeError,
    SettingPersistenceError,
)

# =============================================================================
#  ------------------------------- DgmPt CLASS -------------------------------
# =============================================================================


class PDiagramPoint(tuple):
    def __init__(self, iterable):
        self._set_birth_death_pair(iterable)

        self.x = self.birth
        self.y = self.death

    def _set_birth_death_pair(self, iterable):
        if len(iterable) < 1:
            raise SchroedingersException(
                "Empty homology class foun. "
                "Please provide a time of birth and death! " + str(iterable)
            )

        if len(iterable) < 2:
            raise EndOfTimeError(
                "Everything comes to an end, even homology classes. "
                "Please provide a time of death! " + str(iterable)
            )
        elif len(iterable) > 2:
            raise SchroedingersException(
                "No state beyond birth and death " "implemented yet!" + str(iterable)
            )
        else:
            self.birth = iterable[0]
            self.death = iterable[1]

    # ------------------------------- Properties --------------------------------

    @property
    def birth(self):
        return self._birth

    @birth.setter
    def birth(self, value):
        self._birth = float(value)

    @property
    def death(self):
        return self._death

    @death.setter
    def death(self, value):
        if value < self.birth:
            raise TimeParadoxError(
                "Homology class cannot die before it was born! "
                f"PDiagramPoint = ({self.birth}, {value})"
            )
        self._death = float(value)

    @property
    def persistence(self):
        return self.death - self.birth

    # ---------------------------- Dunder Methods ----------------------------

    def __getitem__(self, key):
        if key in (0, "birth"):
            return self.birth
        elif key in (1, "death"):
            return self.death
        else:
            raise SchroedingersException(
                "No state beyond birth and death " "implemented yet!"
            )

    def __repr__(self):
        return repr((self.birth, self.death))

    def __len__(self):
        return 2

    def __lt__(self, other):
        try:
            return self.persistence < other.persistence
        except AttributeError:
            return self.persistence < other

    def __gt__(self, other):
        try:
            return self.persistence > other.persistence
        except AttributeError:
            return self.persistence > other

    def __le__(self, other):
        try:
            return self.persistence <= other.persistence
        except AttributeError:
            return self.persistence <= other

    def __ge__(self, other):
        try:
            return self.persistence >= other.persistence
        except AttributeError:
            return self.persistence >= other

    def __eq__(self, other):
        return (self.birth == other[0]) and (self.death == other[1])

    def __add__(self, other):
        return type(self)((self.birth + other, self.death + other))

    def __mul__(self, other):
        return type(self)((self.birth * other, self.death * other))

    def __truediv__(self, other):
        return type(self)((self._birth / other, self._death / other))


# =============================================================================
#  -------------------------------- Dgm CLASS --------------------------------
# =============================================================================
class PDiagram(list):
    def __init__(self, iterable=(), dim=None, diameter=None):
        super().extend(sorted(map(PDiagramPoint, iterable), reverse=True))
        if dim is not None:
            self.set_dim(dim)
        if diameter is not None:
            self.set_diameter(diameter)

    @classmethod
    def from_gtda(cls, arr, dim=1, diameter=None):
        dgm = arr[arr[:, 2] == dim][:, :2]
        return cls(dgm, dim=dim, diameter=diameter)

    # -------------------------------- Proerties ---------------------------------
    @property
    def diameter(self):
        return self._diameter

    def set_diameter(self, diameter):
        assert isinstance(diameter, float)
        self._diameter = diameter

    @property
    def dim(self):
        return self._dim

    def set_dim(self, dim):
        assert isinstance(dim, int)
        self._dim = dim

    @property
    def births(self):
        births, deaths = zip(*self)
        return np.array(births)

    @property
    def deaths(self):
        births, deaths = zip(*self)
        return np.array(deaths)

    @property
    def plengths(self):
        if len(self) == 0:
            return np.array([])
        return self.deaths - self.births

    @property
    def pratios(self):
        return self.deaths / self.births

    @property
    def signal(self):
        return self[0].death - self[0].birth

    # ------------------------- Legacy Properties --------------------------
    @property
    def persistences(self):
        warnings.warn(
            "The property `persistences` is depricated! "
            "Please use the method `psequence` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return tuple(pt.persistence for pt in self)

    @property
    def sequence(self, length=None):
        warnings.warn(
            "The property `sequence` is depricated! "
            "Please use the method `psequence` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.signal > 0:
            return tuple(p / self.signal for p in self.persistences)
        else:
            return ()

    @property
    def score(self):
        warnings.warn(
            "The property `score` is depricated! "
            "Please use the method `ring_score` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.ring_score()

    # -------------------------------- Methods ---------------------------------
    def psequence(self, normalisation="signal"):
        """Return 1-dimensional transformation of persistence diagram.

        Parameters
        ----------
        normalisation : str, optional
            Acceptable normalisation: `diameter`, `signal`. By default 'signal'.

        Returns
        -------
        np.array
            Persistence sequence.

        Raises
        ------
        ValueError
            If `normalisation` is not known.
        """
        pseq = self.plengths

        if len(pseq) == 0:
            return pseq
        if normalisation is None:
            return pseq
        elif normalisation.lower() == "diameter":
            return pseq / self.diameter
        elif normalisation.lower() == "signal":
            return pseq / self.signal
        else:
            raise ValueError(f"normalisation `{normalisation}` unknown.")

    def copy(self):
        other = type(self)(self, diameter=self.diameter, dim=self.dim)
        return other

    def append(self, item):
        list.append(self, PDiagramPoint(item))
        self.sort(reverse=True)

    def trimmed(self, length=None):
        if length is None or length == np.inf:
            return self[self > 0]

        if length <= len(self):
            return self[:length]

        else:
            other = self.copy()
            other.extend([(0, 0)] * (length - len(self)))
            return other

    def extend(self, iterable):
        super().extend(type(self)(iterable))
        self.sort(reverse=True)

    def to_array(self):
        return np.array(self)

    # ---------------------------- Function calls -----------------------------
    def ring_score(self, flavour="geometric", nb_pers=None, exponent=2):
        return ring_score_from_sequence(
            self.psequence(), flavour=flavour, nb_pers=nb_pers, exponent=exponent
        )

    def plot(self, ax=None, return_artist_obj=False, **kwargs):
        return plot_dgm(self, ax=ax, return_artist_obj=return_artist_obj, **kwargs)

    # ----------------------------- Dunder Method ------------------------------

    def __getitem__(self, item):
        if isinstance(item, slice):
            return type(self)(
                super().__getitem__(item), diameter=self.diameter, dim=self.dim
            )
        try:
            item_iter = iter(item)
        except TypeError:
            return PDiagramPoint(
                super().__getitem__(item))
        try:
            return type(self)(
                super().__getitem__(item_iter), diameter=self.diameter, dim=self.dim
            )
        except TypeError:
            return type(self)(
                compress(self, item_iter), diameter=self.diameter, dim=self.dim
            )

    def __setitem__(self, index, value):
        raise SettingPersistenceError(
            "Manually setting a persistence point is forbidden. "
            "Use `append` or `extend` instead!"
        )

    def __lt__(self, item):
        return list(pt < item for pt in self)

    def __gt__(self, item):
        return list(pt > item for pt in self)

    def __le__(self, item):
        return list(pt <= item for pt in self)

    def __ge__(self, item):
        return list(pt >= item for pt in self)

    def __str__(self):
        return str(list(self)).replace(", (", ",\n (")

    def __repr__(self):
        return f"{type(self).__name__}({list(self)})"


# =============================================================================
#  ------------------------------ FullDgm CLASS ------------------------------
# =============================================================================


class FullPDgm(MutableMapping):
    def __init__(self, data=()):
        if isinstance(data, (MutableMapping, dict)):
            self.dimensions = tuple(sorted(data.keys()))
            self.mapping = {}
            # needs more checking
            self.update(data)
        if isinstance(data, np.ndarray):
            self.from_numpy_array(data)

        self._sort_homologies()

    # -------------------------------- Methods ---------------------------------

    def from_numpy_array(self, data):
        m, n = data.shape
        assert 3 in {m, n}
        if n != 3:
            data = data.T
        self.dimensions = self._extract_dimensions(data)
        self.mapping = {}
        self.update({key: data[data[:, 2] == key][:, :2] for key in {0, 1}})

    def _extract_dimensions(self, data):
        dimensions = set(data[:, 2])
        assert all(map(float.is_integer, dimensions))
        return tuple(sorted(map(int, dimensions)))

    def _sort_homologies(self):
        pass

    # ----------------------------- Dunder Method ------------------------------

    def __getitem__(self, key):
        return self.mapping[key]

    def __setitem__(self, key, value):
        if key in self:
            del self[self[key]]
        self.mapping[key] = value

    def __delitem__(self, key):
        del self.mapping[key]

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.mapping)

    def __repr__(self):
        return f"{type(self).__name__}({self.mapping.__repr__()})"


# =============================================================================
#  ---------------------------------- Legacy --------------------------------
# =============================================================================


# This code will be removed in future versions
def load_dgm(fname=None, **kwargs):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dgm_array = np.genfromtxt(fname, **kwargs)
        if len(w) == 0:
            return PDiagram(dgm_array)
        elif issubclass(w[-1].category, UserWarning):
            return PDiagram()
        else:
            raise Exception
