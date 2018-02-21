'''
Created on Oct 19, 2017

@author: rene
'''

import pandas as pd

if __name__ == '__main__':
    print(pd.__version__) 
    names = ['student1', 'student2', 'student3', 'student4', 'student5']
    marks_percentage = [80,70,60,90,65]
    studentsDataset = list(zip(names,marks_percentage))
    print str(studentsDataset)
    
    df = pd.DataFrame(data = studentsDataset, columns=['Names', 'marks_percentage'])
    print str(df)
    
    #df.to_csv('student_dataset.csv')
    
    #df1 = pd.read_csv('student_dataset.csv')
    #print str(df1)
    
    print str(df['marks_percentage'].max())
    
    sorted = df.sort_values(['marks_percentage'], ascending=False)
    print str(sorted.head())
    print str(sorted.head(1))
    
    print str(list(df.columns.values))
    
    print str(df.describe())
    