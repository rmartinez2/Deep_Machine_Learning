# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 11:57:38 2018

@author: rene
"""

#Our graph will be a global variable
#The Graph will represent our neural network
#Operation will contain all mathematical operations
#for our graph

#A Placeholder is an empty node that needs an input value
#Variables will be the weights of the graph
#Graph will connect variables and place holders


#We're going to start with a linear regression problem
#z = Ax +b
#Where
#A = 10
#b = 1

#Now that the Graph has all the nodes, we need to execute
#the operations within a Session

#To traverse the graph, we'll use PostOrder Tree Traversal

class Operation():
    
    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.output_nodes = []
        
        for node in self.input_nodes:
            node.output_nodes.append(self)
    
        _default_graph.operations.append(self)
        
    def compute(self):
        pass
    

class add(Operation):
    
    def __init__(self, x, y):
        super().__init__([x,y])
        
    def compute(self, x, y):
        self.inputs = [x, y]
        return x + y
        
class multiply(Operation):
    
    def __init__(self, x, y):
        super().__init__([x,y])
        
    def compute(self, x, y):
        self.inputs = [x, y]
        return x * y
        
class matmul(Operation):
    
    def __init__(self, x, y):
        super().__init__([x,y])
        
    def compute(self, x, y):
        self.inputs = [x, y]
        return x.dot(y)
        

class PlaceHolder():
    
    def __init__(self):
        self.output_nodes = []
        _default_graph.placeholders.append(self)
        
class Variable():
    
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.output_nodes = []
        _default_graph.variables.append(self)
        

class Graph():
    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []
        
    def set_as_default(self):
        global _default_graph
        _default_graph = self

def traverse_postorder(operation):
    """
    PostOrder Traversal of Nodes. Basically makes sure computations are done
    in the correct order (Ax first, then Ax + b)
    """
    nodes_postorder = []
    
    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
            nodes_postorder.append(node)
        
    recurse(operation)
    return nodes_postorder
    
        
class Session():
    
    def run(self, operation, feed_dict={}):
        
        nodes_postorder = traverse_postorder(operation)
        
        for node in nodes_postorder:
            if isinstance(node, PlaceHolder):
                node.output = feed_dict[node]
            
            elif isinstance(node, Variable):
                node.output = node.value
            
            else:
                #Operation
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)
                
            if isinstance(node.output, list):
                node.output = np.array(node.output)
                
        return operation.output
            
        