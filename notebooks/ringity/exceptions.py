#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 13:42:53 2018

@author: myoussef
"""

class RingityException(Exception):
    pass

# ---------------------- Ripser Exceptions ----------------------------

class RipserOutputError(Exception):
    pass 


# ---------------------- Diagram Exceptions ----------------------------
class DiagramException(RingityException):
    pass

class SchroedingersException(IndexError):
    pass 

class TimeParadoxError(DiagramException):
    pass

class BeginningOfTimeError(DiagramException):
    pass

# ---------------------- Graph Type Exceptions -----------------------------

class GraphTypeError(Exception):
    pass

    
class DigraphError(GraphTypeError):
    pass
class UnknownGraphType(GraphTypeError):
    pass
class MultigraphError(GraphTypeError):
    pass
