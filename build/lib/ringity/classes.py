
from itertools import starmap, islice
from ringity.methods import draw_diagram
from ringity.exceptions import SchroedingersException, TimeParadoxError, BeginningOfTimeError

import warnings
import numpy as np


"""
===============================================================================
 _______  _______ _________          _______  ______   _______
(       )(  ____ \\__   __/|\     /|(  ___  )(  __  \ (  ____ \
| () () || (    \/   ) (   | )   ( || (   ) || (  \  )| (    \/
| || || || (__       | |   | (___) || |   | || |   ) || (_____
| |(_)| ||  __)      | |   |  ___  || |   | || |   | |(_____  )
| |   | || (         | |   | (   ) || |   | || |   ) |      ) |
| )   ( || (____/\   | |   | )   ( || (___) || (__/  )/\____) |
|/     \|(_______/   )_(   |/     \|(_______)(______/ \_______)
===============================================================================
"""

def score(iterable, sort=True, **kwargs):
    if sort:
        iterable = sorted(iterable, **kwargs, reverse=True)
    iterator = iter(iterable)
    try:
        signal = next(iterator)
        return 1-sum((noise/signal) / 2**i for i,noise in enumerate(iterator,1))
    except StopIteration:
        return 0

def random_DgmPt(lb=0, ub=1):
    a, b = np.random.uniform(lb,ub,size=2)
    return DgmPt(min(a,b), max(b,a))

def random_Dgm(lb=0, ub=1, length=1):
    return Dgm([random_DgmPt(lb,ub) for _ in range(length)])

def save_dgm(dgm, fname, **kwargs):
    array = np.array([(k.birth, k.death) for k in dgm])
    np.savetxt(fname, array, **kwargs)


def load_dgm(fname=None, **kwargs):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dgm_array = np.genfromtxt(fname, **kwargs)
        if len(w) == 0:
            return Dgm(dgm_array)
        elif issubclass(w[-1].category, UserWarning):
            return Dgm()
        else:
            raise Exception


def indexify_dgm(dgm, inplace=False):
    values = set.union(*map(set,dgm))
    value2index = {value:index for index,value in enumerate(sorted(values))}
    if inplace:
        dgm._values = [DgmPt(*map(value2index.get,pt)) for pt in dgm]
    else:
        return Dgm(DgmPt(*map(value2index.get,pt)) for pt in dgm)


"""
===============================================================================
 _______  _        _______  _______  _______  _______  _______
(  ____ \( \      (  ___  )(  ____ \(  ____ \(  ____ \(  ____ \
| (    \/| (      | (   ) || (    \/| (    \/| (    \/| (    \/
| |      | |      | (___) || (_____ | (_____ | (__    | (_____
| |      | |      |  ___  |(_____  )(_____  )|  __)   (_____  )
| |      | |      | (   ) |      ) |      ) || (            ) |
| (____/\| (____/\| )   ( |/\____) |/\____) || (____/\/\____) |
(_______/(_______/|/     \|\_______)\_______)(_______/\_______)
===============================================================================
"""

class DgmPt():
    def __init__(self, birth=0, death=np.inf):
        self._birth = birth
        self._death = death
        if birth > death:
            raise TimeParadoxError('Homology class cannot die before it was born! '
                              f'({birth}, {death})')
        if birth < 0:
            raise BeginningOfTimeError('Hole must be born after '
                                       f'beginning of time! {birth}')
    def __getitem__(self, key):
        if   key == 0:
            return self.birth
        elif key == 1:
            return self.death
        else:
            raise SchroedingersException('No state beyond birth and death '
                                         'implemented yet!')
    def __repr__(self):
        return repr((self.birth, self.death))
    def __len__(self):
        return 2

    def __lt__(self, other):
        return self.persistence <  other.persistence
    def __le__(self, other):
        return self.persistence <= other.persistence
    def __eq__(self, other):
        return (self.birth == other[0]) and (self.death == other[1])

    def __add__(self, other):
        return DgmPt(self.birth + other, self.death + other)
    def __truediv__(self, other):
        return DgmPt(self._birth/other, self._death/other)


    @property
    def birth(self):
        return self._birth
    @property
    def death(self):
        return self._death
    @property
    def persistence(self):
        return self.death - self.birth

    @property
    def x(self):
        return self._birth
    @property
    def y(self):
        return self._death



class Dgm():
    def __init__(self, pts=[]):
        if isinstance(pts,np.ndarray):
            m, n = pts.shape
            assert 2 in {m,n}
            if n != 2:
                pts = pts.T

        self._values = sorted(starmap(DgmPt,pts),
                              key=lambda pt:pt.persistence,
                              reverse=True)

    def __getitem__(self, item):
        return self.values[item]
    def __repr__(self):
        return f'Dgm({repr(self.values)})'
    def __str__(self):
        return '\n'.join(str(pt) for pt in self.values)
    def __len__(self):
        return len(self.values)
    def __eq__(self, other):
        return self.values == other.values
    def __add__(self, other):
        return Dgm(self.values+other.values)
    def __truediv__(self, other):
        return Dgm(pt/other for pt in self)


    @property
    def values(self):
        return self._values
    @property
    def births(self):
        return [pt.birth for pt in self._values]
    @property
    def deaths(self):
        return [pt.death for pt in self._values]

    @property
    def signal(self):
        return self[0].persistence
    @property
    def sequence(self):
        return (pt.persistence/self.signal for pt in self)
    @property
    def gap(self):
        return self.signal - self[1].persistence
    @property
    def score(self, max_len = 50):
        if len(self) == 0:
            return 0
        else:
            iter_sequence = islice(enumerate(self.sequence), 1, max_len)
            return 1-sum(pers/2**i for i,pers in iter_sequence)


    def save(self, fname, **kwargs):
        save_dgm(self.values, fname, **kwargs)

    def append(self, pt):
        dgm_pt = DgmPt(*pt)
        self._values.append(dgm_pt)
        self._values = sorted(self._values,
                              key=lambda pt:pt.persistence,
                              reverse=True)

    def cap(self, n, inplace=False):
        if inplace:
            self._values = self.values[:n]
        else:
            return Dgm(self.values[:n])

    def crop(self, n, inplace=False):
        if inplace:
            dgm = self
        else:
            dgm = self.copy()
        for i in range(n):
            dgm.append((0,0))
        return dgm.cap(n, inplace=False)


    def copy(self, index=False):
        dgm = eval(repr(self))
        if index:
            dgm = indexify_dgm(dgm)
        return dgm

    def draw(self, ax=None):
        draw_diagram(self, ax=ax)
