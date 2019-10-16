#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 14:24:33 2018

@author: myoussef
"""

_ringity_parameters = {
    'verbose': False,  # defines the verbose default
    'no'     : {'n', 'no', 'nope', 'nah',  'negative', 'false'},
    'yes'    : {'y', 'yes', 'yep', 'yeah', 'positive', 'true'},
    }

_assertion_statement =  'This should never happen, but apparently it does. Please contact mk.youssef@hotmail.com if you encounter this in the wild. Thanks!'


def _make_options_dict(verbose=False):
    """ Make a dictionary out of the non-None arguments. """
    options = {k: v for k, v in locals().items() if v is not None}
    return options


def set_parameters(verbose=False):
    """
    Set parameters.

    Parameters
    ----------
    vebose : bool, optional
        If True, something something.  
        If False, something something.
        The default is False.
    

    Examples
    --------
    Verbose toggle can be set:

    >>> rng.set_parameters(verbose=False)
    >>> print('something something')
    'something something'
    """

    opt = _make_options_dict(verbose)
    _ringity_parameters.update(opt)


def get_printoptions():
    """
    Return the current ringity parameters.

    Returns
    -------
    _ringity_parameters : dict
        Dictionary of current ringity parameters with keys

          - verbose : bool

        For a full description of these options, see `set_printoptions`.
    """
    return _ringity_parameters.copy()
