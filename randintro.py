##import random
##for i in range(5):
##    a=random.random()#it generates floating points from 0 to 1
##    print(a)
##    b=random.uniform(10,20)#it generates random floating points in given range
##    print(b)
##    c=random.randint(10,20)#it generates random integer values
##    print(c)
##    d=random.randrange(1,10)#it generates random values in given range
##    print(d)
##    e=random.randrange(1,10,2)#it generates random values with step
####    print(e)
##l=['swathi','nagu','siva','divya']
##for i in range(5):
##    print(random.choice(l)) #it generates random 
##    print(random.sample(l,2)) #it generates sub list in given range
##random.shuffle(l)#it shuffles the list
##print(l)
import numpy as np
#a=np.array([1,2,3])
#print(a)
##a=np.array([[1,2,3],[5,4,3]])
##print(a)
##a=np.arange(5) # to generate sequential numbers
##print(a)
##a=np.arange(5,15,2)# it generates sequential numbers with step
##print(a)
##a=np.zeros((3,2))#it generates 0 in given order
##print(a)
##a=np.ones((2,3))#it generates 1 in given order
##print(a)
##a=np.empty((2,3))
##print(a)
##a=np.arange(50,60)
##print(a[a>55])


##a=np.arange(1,25).reshape(6,4)
##print(a)
##a=np.arange(1,25).reshape(3,4,2)
##print(a)
##print(a.T)


##a=np.arange(1,10).reshape(3,3)#to convert 1 dimensional to multidimensional
##print(a)
##print(a.T)#to transpose a matrix
##a=np.arange(1,10).reshape(3,3)
##print(a)
##a=a.flatten(order='C')# to convert multidimensional to one dimensional
##print(a)              # C means to generate row wise values in a matrix
##
##a=np.arange(1,10).reshape(3,3)
##print(a)
##a=a.flatten(order='F')# to convert multidimensional to one dimensional
##print(a)#F means to generate column wise values in a matrix



#***append in numpy***
##append the values in rowwise
##a=np.array([[1,2,3],[4,5,6]])
##print(a)
##b=np.append(a,[[8,9,10]],axis=0)#axis=0 means we have to add the new row in row wise
##print(b)                               # append the values at the end of array
#append the values in column wise
##a=np.array([[5,6,7],[8,9,10]])
##print(a)
##b=np.append(a,[[12],[15]],axis=1)# axis=1 means we have to add values in column wise
##print(b)


#*** insert the elements at any position ***
##a=np.arange(1,17).reshape(4,4)
##print(a)
##b=np.insert(a,2,[[17,18,19,20]],axis=0) # row wise in specified position
##print(b)
##
##a=np.arange(1,17).reshape(4,4)
##print(a)
##b=np.insert(a,1,[[17,18,19,20]],axis=1)#column wise in specified position
##print(b)


##split function in numpy
##for 1-dimensional
##a=np.arange(1,9)
##print(a)
##b=np.split(a,4)# it divides the array into sub parts of array,accoding to size of array for that we give proper size 
##print(b)
##a=np.arange(1,9)
##print(a)
##b=np.split(a,[2,3])
##print(b)


##*** for 2 dimensional splitting****
##a=np.arange(1,17).reshape(4,4)
##print(a)
##b=np.split(a,2,axis=0) # it splits the matrix after specified row number is mentioned
##print(b)
##
##a=np.arange(1,17).reshape(4,4)
##print(a)
##b=np.split(a,2,axis=1) # it splits the matrix after specified column number is mentioned
##print(b)


## ***delete in numpy***
##a=np.arange(1,17).reshape(4,4)
##print(a)
##b=np.delete(a,1,axis=0) # it deletes the specified row in a matrix 1 means position of row
##print(b)

##a=np.arange(1,17).reshape(4,4)
##print(a)
##b=np.delete(a,1,axis=1) #it deletes the specified column in a matrix 1 means position of column
##print(b)

##****slicing and dicing****
a=np.arange(1,17).reshape(4,4)
print(a)
print(a[1: ,2: ]) #it gives output as excluding 0th row and 0th and 1st column

a=np.arange(1,17).reshape(4,4)
print(a)
print(a[2: ,1: ])# it gives output as excluding 0th,1st row and 0th column






    

