'''
Created on Jan 1, 2018

@author: rene
'''
import urllib2
import sys


if __name__ == '__main__':

    target_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")
    data = urllib2.urlopen(target_url)
    
    x_list = []
    labels = []
    
    for line in data:
        #split on comma
        row = line.strip().split(",")
        x_list.append(row)
        
    
    #sys.stdout.write("Number of Rows of Data = " + str(len(x_list)) + '\n')
    #sys.stdout.write("Number of Columns of Data = " + str(len(x_list[1])) + '\n')
    
    n_row = len(x_list)
    n_col = len(x_list[1])
    
    type = [0]*3
    col_counts = []
    
    for col in range(n_col):
        for row in x_list:
            try:
                a = float(row[col])
                
                if isinstance(a, float):
                    type[0] += 1
                
            except ValueError:
                if (len(row[col])) > 0:
                    type[1] += 1
                else:
                    type[2] += 1
    
    col_counts.append(type)
    type = [0]*3
    
    sys.stdout.write("Col#" + '\t' + "Number" + '\t' + "Strings" + '\t' + "Other\n")
    
    i_col = 0
    
    for types in col_counts:
        sys.stdout.write(str(i_col) + '\t\t' + str(types[0]) + '\t\t' + str(types[1]) + '\t\t' + str(types[2]) + "\n")
        i_col += 1

    
    
    