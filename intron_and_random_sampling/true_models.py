import numpy as np

from bingo.symbolic_regression.agraph.agraph import AGraph

def create_complex_true_model():
    '''
    This models complexity is 9.
    y(x) = sin(p0*x) + x*x + p1*x
    '''
    true_model_ca = np.array([[0,0,0],
                              [1,0,0],
                              [1,1,1],
                              [4,1,0],
                              [4,2,0],
                              [6,3,3],
                              [4,0,0],
                              [2,5,6],
                              [2,7,4]])

    true_model = AGraph()
    true_model.command_array = true_model_ca

    return true_model

def create_simple_true_model():
    '''
    This models complexity is 9.
    y(x) = sin(x) + p0*x
    '''
    true_model_ca = np.array([[0,0,0],
                              [1,0,0],
                              [6,0,0],
                              [4,1,0],
                              [2,2,3]])

    true_model = AGraph()
    true_model.command_array = true_model_ca

    return true_model
