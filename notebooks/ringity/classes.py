"""
 __       __             __      __                        __           
|  \     /  \           |  \    |  \                      |  \          
| $$\   /  $$  ______  _| $$_   | $$____    ______    ____| $$  _______ 
| $$$\ /  $$$ /      \|   $$ \  | $$    \  /      \  /      $$ /       \
| $$$$\  $$$$|  $$$$$$\\$$$$$$  | $$$$$$$\|  $$$$$$\|  $$$$$$$|  $$$$$$$
| $$\$$ $$ $$| $$    $$ | $$ __ | $$  | $$| $$  | $$| $$  | $$ \$$    \ 
| $$ \$$$| $$| $$$$$$$$ | $$|  \| $$  | $$| $$__/ $$| $$__| $$ _\$$$$$$\
| $$  \$ | $$ \$$     \  \$$  $$| $$  | $$ \$$    $$ \$$    $$|       $$
 \$$      \$$  \$$$$$$$   \$$$$  \$$   \$$  \$$$$$$   \$$$$$$$ \$$$$$$$ 
                                                                        
"""


from ringity.methods import draw_diagram
from ringity.exceptions import SchroedingersException, TimeParadoxError, BeginningOfTimeError

import warnings
import numpy as np



def save_dgm(dgm, fname=None, **kwargs):
    array = np.array([[k.birth, k.death] for k in dgm])
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
        
        
def barcode(dgm):
    return [pt.death-pt.birth for pt in dgm]
    

def normalize(dgm):
    p_max = max(barcode(dgm))
    return dgm/p_max


def sorted_barcode(dgm):
    births, deaths = map(np.array,zip(*[(pt.birth,pt.death) for pt in dgm]))
    return np.sort(deaths-births)[::-1]


def normalized_barcode(dgm):
    barcode = sorted_barcode(dgm)
    return barcode / barcode[0]


def get_GGS(dgm):
    if len(dgm) == 0:
        return 0
    elif len(dgm) == 1:
        return 1        
    else:
        noba = normalized_barcode(dgm)
        return 1 - sum([noba[i]/2**i for i in range(1, min(len(noba),55))])
    

def get_hgap(dgm):    
    if len(dgm) == 0:
        return 0
    elif len(dgm) == 1:
        return 1
    else:
        noba = normalized_barcode(dgm)
        p1 = noba[1]
        return 1 - p1
    
    
def get_NSR(dgm):    
    if len(dgm) == 0:
        return 0
    elif len(dgm) == 1:
        return 1        
    else:
        noba = normalized_barcode(dgm)
        return 1 - np.mean(noba[1:])
    

def get_clustering(dgm):
    if len(dgm) == 0:
        return 0
    elif len(dgm) == 1:
        return 1 
    else:
        data = normalized_barcode(dgm)
        score = np.zeros(len(data)-1)
        
        for g in range(1,len(data)):
            data1 = data[:g]
            data2 = data[g:]
        
            mu  = np.mean(data)
            mu1 = np.mean(data1)
            mu2 = np.mean(data2)
        
            TSS = sum((data-mu)**2)
            BSS = len(data1)*(mu1 - mu)**2 + len(data2)*(mu2 - mu)**2
            score[g-1] = BSS/TSS
    return score
    

def indexify_dgm(dgm):
    x_values = set([pt.birth for pt in dgm])
    y_values = set([pt.death for pt in dgm])
    values = x_values.union(y_values)
    indexDict = {value:index for index,value in enumerate(sorted(values))}

    for i,pt in enumerate(dgm):
        dgm[i]._birth = indexDict[pt.birth]
        dgm[i]._death = indexDict[pt.death]
    
    return dgm
    
    
"""
  ______   __                                                   
 /      \ |  \                                                  
|  $$$$$$\| $$  ______    _______   _______   ______    _______ 
| $$   \$$| $$ |      \  /       \ /       \ /      \  /       \
| $$      | $$  \$$$$$$\|  $$$$$$$|  $$$$$$$|  $$$$$$\|  $$$$$$$
| $$   __ | $$ /      $$ \$$    \  \$$    \ | $$    $$ \$$    \ 
| $$__/  \| $$|  $$$$$$$ _\$$$$$$\ _\$$$$$$\| $$$$$$$$ _\$$$$$$\
 \$$    $$| $$ \$$    $$|       $$|       $$ \$$     \|       $$
  \$$$$$$  \$$  \$$$$$$$ \$$$$$$$  \$$$$$$$   \$$$$$$$ \$$$$$$$ 
                                                                                                                               
"""

        
    
class DgmPt():       
    def __init__(self, birth=0, death=np.inf):
        self._birth = birth
        self._death = death
        if birth > death:
            raise TimeParadoxError('Hole cannot die before it was born! ' 
                              f'({birth}, {death})')
        if birth < 0:
            raise BeginningOfTimeError('Hole must be born after the '
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
        return (self.death-self.birth) <  (other.death-other.birth)
    def __le__(self, other):
        return (self.death-self.birth) <= (other.death-other.birth)
    def __eq__(self, other):
        return (self._birth == other[0]) and (self._death == other[1])
    
    def __add__(self, other):
        return DgmPt(self._birth + other._birth, self._death + other._death)
    def __truediv__(self, other):
        return DgmPt(self._birth/other, self._death/other)
              
        
    @property
    def birth(self):
        return self._birth
    @property
    def death(self):
        return self._death
    
    @property
    def x(self):
        return self._birth
    @property
    def y(self):
        return self._death
    
           
        
class Dgm():
    def __init__(self, pts=[]):
        if isinstance(pts,np.ndarray):
            pts.shape = -1,2
        pts_list = [DgmPt(pt[0],pt[1]) for pt in pts]
        self._pts = pts_list 
    def __getitem__(self, item):
        return self._pts[item]        
    def __repr__(self):
        return f'Dgm({repr(self._pts)})'
    def __str__(self):
        return '\n'.join([str(pt) for pt in self._pts])
    def __len__(self):
        return len(self._pts)
    def __eq__(self, other):
        return self._pts == other._pts
    
    def __add__(self, other):
        if len(self) < len(other):
            dgm1 = self
            dgm2 = other
        else:
            dgm1 = other
            dgm2 = self
        
        for i in range(len(dgm2)-len(dgm1)):
            dgm1.append((0,0))
            
        dgm1 = sorted(dgm1, reverse=True)
        dgm2 = sorted(dgm2, reverse=True)
            
        return Dgm([pt1+pt2 for (pt1,pt2) in zip(dgm1,dgm2)])
    
    def __truediv__(self, other):
        return Dgm(pt/other for pt in self)
        
    
    @property
    def GGS(self):
        return get_GGS(self)    
    @property
    def hgap(self):
        return get_hgap(self)
    @property
    def NSR(self):
        return get_NSR(self)
    @property
    def clustering(self):
        return get_clustering(self)
    
      
    def save(self, fname=None):
        save_dgm(self._pts, fname=fname)        
        
    def load(self, name=None, location = '.'):
        return load_dgm(location=location, name=name)
    
    def append(self, pt):
        if len(pt) != 2:
            raise SchroedingersException('No state beyond birth and death '
                                         f'implemented yet! {len(pt)}')
        dgmPt = DgmPt(*pt)
        self._pts.append(dgmPt)
        
    def cap(self, n):
        """Returns a sorted diagram of length n 
        with most persistent diagram points."""
        return sorted(self, reverse=True)[:n]
            
    def copy(self, index=False):
        dgm = eval(repr(self))
        if index:
            dgm = indexify_dgm(dgm)
        return dgm
        
    def draw(self, ax=None):
        draw_diagram(self, ax=ax)
        