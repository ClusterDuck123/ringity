import os
import json
import pandas  as pd

import sys

def main():
    
    df = concatenate_json_folders(sys.argv[2:])
    filename = sys.argv[1]
    df.to_csv(f"{filename}.csv")

    
def concatenate_json_folders(folders):
    
    out = []
    for folder in folders:
        flattened_data = collect_and_flatten_json_files(folder)
        out.append(pd.DataFrame(flattened_data))
    
    return pd.concat(out)
    
def incorporate_json_values(out, raw_json, prefix = None):
    """_summary_

    Args:
        out (dict): A dictionary that cumulatively holds a flattened version of the json file
        raw_json (dict): a dict that holds part of the raw (and still possibly hierachical) json file
        prefix (str): The prefix to the key names in the flattened version, necessary to avoid redundant names. Defaults to None, which triggers substitution with the empty string.
    """

    # spiritually, lists are just dictionaries with integer keys
    if type(raw_json) == list:
        raw_json = dict(enumerate(raw_json))

    # Now, until we're at the lowest level, raw_json will be a dict
    # so keep iteratively unpacking
    if type(raw_json) == dict:
        
        # ensure there are no leading or trailing 
        # underscores in the key
        if prefix == None:
            prefix = ""
        else:
            prefix = prefix + "_"
        
        for i,v in raw_json.items():
            incorporate_json_values(out, v, prefix=f"{prefix}{i}")
    
        
    else: # if neither a list or dict, it's a value
        assert type(raw_json) in [float,int,str]
        
        out[prefix] = raw_json
        
    
        
    
def collect_and_flatten_json_files(root_folder):
    """
    Collects JSON files from the given root folder and its subdirectories, 
    reads them into memory, flattens them into a single dictionary with renamed keys, 
    and raises an error if any JSON file contains a list.

    Args:
        root_folder (str): Path to the root folder to start searching for JSON files.

    Returns:
        list: A list of flattened dictionaries containing data from each JSON files.

    Raises:
        ValueError: If any JSON file contains a list.
    """
    flattened_data = []
    
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(".json"):
                fullpath = os.path.join(dirpath,filename)
                raw_json = json.load(open(fullpath))
                
                out = {}
                incorporate_json_values(out,raw_json)
                
                out["folder"] = fullpath
                flattened_data.append(out)
    return flattened_data
                
main()
