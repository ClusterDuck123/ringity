#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 13:42:53 2018

@author: myoussef
"""

# Base class
class RingityException(Exception):
    pass


# ----------------------------- Dgm Exceptions -----------------------------
class SchroedingersException(IndexError):
    # Don't know yet how to combine two exception classes
    pass
class DgmException(RingityException):
    pass
class TimeParadoxError(DgmException):
    pass
class BeginningOfTimeError(DgmException):
    pass

# ----------------------- NetworkX related Exceptions -----------------------
# Don't know yet how to combine two exception classes
class GraphTypeError(RingityException):
    pass
class DigraphError(GraphTypeError):
    pass
class UnknownGraphType(GraphTypeError):
    pass
class MultigraphError(GraphTypeError):
    pass
class DisconnectedGraphError(RingityException):
    pass
