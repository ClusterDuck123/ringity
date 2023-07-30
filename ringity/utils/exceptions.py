#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 13:42:53 2018

@author: myoussef
"""

# Base class
class RingityException(Exception):
    pass
    
class NotImplementedYetError(RingityException):
    pass

# ---------------------------- Distribution Errors ----------------------------
class DistributionParameterError(RingityException):
    pass
    
# -------------------------- Network Parameter Errors --------------------------
class NetworkParameterError(RingityException):
    pass

class ConflictingParametersError(NetworkParameterError):
    pass

class ProvideParameterError(NetworkParameterError):
    pass


# ----------------------------- Dgm Exceptions -----------------------------
class DgmException(RingityException):
    pass
    
class DgmTimeError(DgmException):
    pass
    
class SchroedingersException(IndexError, DgmTimeError):
    # Don't know yet how to combine two exception classes
    pass   
     
class TimeParadoxError(DgmTimeError):
    pass
    
class BeginningOfTimeError(DgmTimeError):
    pass
    
class EndOfTimeError(DgmTimeError):
    pass
    
class SettingPersistenceError(DgmException):
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
