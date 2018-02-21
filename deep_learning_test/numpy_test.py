'''
Created on Oct 19, 2017

@author: rene
'''
import numpy as np

if __name__ == '__main__':
    print(np.__version__)
    int_array = np.array([2,3,4])
    float_array = np.array([1.2,3.5,5.1])
    print str(int_array)
    print str(float_array)
    
    print str(np.zeros([2,3]))
    print str(np.ones([3,3]))
    
    print str(np.arange(1,10,1))
    
    even_numbers = np.array([2,4,6,8,10])
    odd_numbers = np.array([1,3,5,7,9])
    add_two_arrays = even_numbers + odd_numbers
    sub_two_arrays = even_numbers - odd_numbers
    product_two_arrays = even_numbers * odd_numbers
    print str(add_two_arrays)
    print str(sub_two_arrays)
    print str(product_two_arrays)
    
    matrix_1 = np.array([[1,1],[2,3]])
    matrix_2 = np.array([[2,0], [1,6]])
    print str(matrix_1) + " " + str(matrix_2)
    matrix_multiplication = matrix_1.dot(matrix_2)
    print str(matrix_multiplication)
    
    print str(even_numbers.sum())
    print str(even_numbers.mean())
    print str(even_numbers.max())
    