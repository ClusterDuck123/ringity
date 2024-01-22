import json
import numpy as np

from collections import defaultdict
from ringity.tda.pdiagram.pdiagram import PDiagram

def read_pdiagram(fname, **kwargs):
    """
    Wrapper for numpy.genfromtxt.
    """
    return PDiagram(np.genfromtxt(fname, **kwargs))
    
def write_pdiagram(dgm, fname, **kwargs):
    """
    Wrapper for numpy.savetxt.
    """
    array = np.array(dgm)
    np.savetxt(fname, array, **kwargs)

def json_to_dictoflist(fname, 
            lambda_key = lambda x:x,
            new = False):
    """Takes a json file and turns it into PDiagrams in a dict-of-list object. 
    This can be useful for data where the inner most level (the list) are samples depending 
    on one parameters (key of the dict)."""
    data_dict = defaultdict(list)
    if new:
        return data_dict
    if fname.is_file():
        with open(fname, 'r') as f:
            json_dict = json.load(f)
            for jkey in json_dict:
                dkey = lambda_key(jkey)
                data_dict[dkey] = list(map(PDiagram, json_dict[jkey]))
    return data_dict

def json_to_dictofdictoflist(fname, 
            lambda_key1 = lambda x:x, 
            lambda_key2 = lambda x:x,
            new = False):
    """Takes a json file and turns it into PDiagrams in a dict-of-dict-of-list object. 
    This can be useful for data where the inner most level (the list) are samples depending 
    on two other parameters (key1 and key2)."""
    data_dict = defaultdict(lambda: defaultdict(list))
    if new:
        return data_dict
    if fname.is_file():
        with open(fname, 'r') as f:
            json_dict = json.load(f)
            for jkey1 in json_dict:
                dkey1 = lambda_key1(jkey1)
                for jkey2 in json_dict[dkey1]:
                    dkey2 = lambda_key2(jkey2)
                    data_dict[dkey1][dkey2] = list(map(PDiagram, json_dict[jkey1][jkey2]))
    return data_dict

def data_to_json(fname, data):
    """Writes data into json file."""
    data_json = json.dumps(data)
    with open(fname, 'w') as f:
        f.write(data_json)