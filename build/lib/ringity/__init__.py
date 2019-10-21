#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 17:28:12 2018

@author: myoussef
"""


name = "ringity"
__author__ = "Markus Kirolos Youssef"
__version__ = "0.0a3"

from ringity.methods   import *
from ringity.classes   import *
from ringity.core      import *
from ringity.centralities import *
from ringity.constants import _assertion_statement

import os
import sys
import shutil
import warnings
import subprocess
import numpy as np


RINGITY_PATH = os.path.dirname(__file__)

# Check for proper Python version
if sys.version_info[:2] < (3, 6):
    m = "Python 3.6 or later is required for ringity (%d.%d detected)."
    raise ImportError(m % sys.version_info[:2])

def install_ripser():
    try:
        command = f'cd {RINGITY_PATH}/ripser && make'
        subprocess.run(command, shell=True)
    except:
        print('An unnkown error occured while trying to install ripser with:\n'
              + ' '.join(command), sys.exc_info()[0])
        raise

    # Built-in testrun
    try:
        command = [f'{RINGITY_PATH}/ripser/ripser',
                   f'{RINGITY_PATH}/ripser/examples/sphere_3_192.lower_distance_matrix',]
        subprocess.run(command)
        print('Ripser successfully installed!')
    except:
        print('An unnkown error occured while trying to test ripser with:\n'
              + ' '.join(command), sys.exc_info()[0])
        raise
    test_ripser()


def test_ripser(verbose=False):
    if not os.path.isfile(f"{RINGITY_PATH}/ripser/ripser"):
        install_ripser()
    else:
        # create dummy .csv file
        np.savetxt(f'{RINGITY_PATH}/ripser/foo.csv', np.zeros(1), fmt='%0.0f')

        # run ripser
        command = [f'{RINGITY_PATH}/ripser/ripser', f'{RINGITY_PATH}/ripser/foo.csv']
        try:
            completed_process = subprocess.run(command, stdout=subprocess.PIPE)
            output = completed_process.stdout.decode("utf-8")
            if 'distance matrix with 1 points\n' not in output:
                warnings.warn('Somehow ripser is not working properly!')
                print(f"'distance matrix with 1 points\n' not in '{output}'")

        except FileNotFoundError:
            print('Ripser seems to not be installed properly. Ringity is '
                  'trying to reinstall it.')
            try:
                install_ripser()
            except:
                answer = input('Intallation failed. Is ringity allowed to '
                               'delete the directory \\ripser and try to '
                               'download ripser again? (y/n) \n')
                answer = yes_or_no(answer)


def yes_or_no(answer):
    while answer not in ['y','n',]:
        answer = input("Please only type in 'y' or 'n'! ")

    if answer == 'y':
        shutil.rmtree(f'{RINGITY_PATH}/ripser')
        download_ripser()
        install_ripser()
    elif answer == 'n':
        print('Okay... please make sure ripser is working properly yourself.')
    else:
        assert False, _assertion_statement


# ------------------------  script starts here  -------------------------------
test_ripser()
