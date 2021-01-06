# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # **Assignment For Numpy**
# %% [markdown]
# Difficulty Level **Beginner**
# %% [markdown]
# 1. Import the numpy package under the name np

# %%
import numpy as np

# %% [markdown]
# 2. Create a null vector of size 10 

# %%
a=np.empty(10)
a


# %%
3. Create a vector with values ranging from 10 to 49


# %%
b=np.arange(10,49)
b

# %% [markdown]
# 4. Find the shape of previous array in question 3

# %%
b.shape

# %% [markdown]
# 5. Print the type of the previous array in question 3

# %%
b.dtype

# %% [markdown]
# 6. Print the numpy version and the configuration
# 

# %%
print(np.__version__)
print(np.show_config())

# %% [markdown]
# 7. Print the dimension of the array in question 3
# 

# %%
b.ndim

# %% [markdown]
# 8. Create a boolean array with all the True values

# %%
np.full((3,3),True,dtype=bool)

# %% [markdown]
# 9. Create a two dimensional array
# 
# 
# 

# %%
np.arange(12).reshape(3,4)

# %% [markdown]
# 10. Create a three dimensional array
# 
# 

# %%
np.arange(12).reshape(3,2,2)

# %% [markdown]
# Difficulty Level **Easy**
# %% [markdown]
# 11. Reverse a vector (first element becomes last)

# %%
a=np.arange(10)
a[::-1]

# %% [markdown]
# 12. Create a null vector of size 10 but the fifth value which is 1 

# %%


# %% [markdown]
# 13. Create a 3x3 identity matrix

# %%
np.identity(3)

# %% [markdown]
# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# %%
arr = np.array([1,2,3,4,5])
arr=arr.astype(float)
arr.dtype

# %% [markdown]
# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# %%
arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]])  
              
arr2 = np.array([[0., 4., 1.],

           [7., 2., 12.]])
arr1*arr2

# %% [markdown]
# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# %%
arr1 = np.array([1., 2., 3., 4., 5., 6.]) 
arr2 = np.array([0., 4., 1., 7., 2., 12.])
compare=arr1==arr2
compare

# %% [markdown]
# 17. Extract all odd numbers from arr with values(0-9)

# %%
arr=np.arange(10)
arr[arr%2==1]

# %% [markdown]
# 18. Replace all odd numbers to -1 from previous array

# %%
arr[arr%2==1]=-1
arr

# %% [markdown]
# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# %%
arr=np.arange(10)
arr
arr[5:9]=12
arr

# %% [markdown]
# 20. Create a 2d array with 1 on the border and 0 inside

# %%
arr=np.ones((5,5))
arr[1:-1,1:-1]=0
arr

# %% [markdown]
# Difficulty Level **Medium**
# %% [markdown]
# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# %%
arr2d = np.array([[1, 2, 3],

            [4, 5, 6], 

            [7, 8, 9]])
arr2d[arr2d==5]=12
arr2d

# %% [markdown]
# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# %%
arr3d = np.array([[[1, 2, 3],[4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[0:1]=64
arr3d

# %% [markdown]
# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# %%
arr2d=np.arange(10).reshape(2,5)
arr2d[0:1]

# %% [markdown]
# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# %%
arr2d=np.arange(10).reshape(2,5)
arr2d[1,1]

# %% [markdown]
# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# %%
arr2d=np.arange(10).reshape(2,5)
arr2d[:,2]

# %% [markdown]
# 26. Create a 10x10 array with random values and find the minimum and maximum values

# %%
arr=np.random.rand(10,10)
arr.max()
arr.min()

# %% [markdown]
# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# %%
a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.intersect1d(a,b)

# %% [markdown]
# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# %%
a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.where(a==b)

# %% [markdown]
# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# %%
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
data[names!='Will']

# %% [markdown]
# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# %%
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
data[(names!='Will')&(names!='Joe')]

# %% [markdown]
# Difficulty Level **Hard**
# %% [markdown]
# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# %%
arr=np.arange(1,16).reshape(5,3)
arr

# %% [markdown]
# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# %%
arr=np.arange(1,17).reshape(2,2,4)
arr

# %% [markdown]
# 33. Swap axes of the array you created in Question 32

# %%
arr.T

# %% [markdown]
# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# %%
arr=np.arange(10)
arr=np.sqrt(arr)
arr[arr < 0.5]=0
arr

# %% [markdown]
# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# %%
arr1=np.random.rand(12)
arr2=np.random.rand(12)
arr3=np.maximum(arr1,arr2)
arr3

# %% [markdown]
# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# %%
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)

# %% [markdown]
# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# %%
a = np.array([1,2,3,4,5]) 
b = np.array([5,6,7,8,9])
np.setdiff1d(a,b)

# %% [markdown]
# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# %%
sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
newColumn = np.array([10,10,10])
sampleArray[:,1]=newColumn
sampleArray

# %% [markdown]
# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# %%
x = np.array([[1., 2., 3.], [4., 5., 6.]]) 
y = np.array([[6., 23.], [-1, 7], [8, 9]])
x.dot(y)

# %% [markdown]
# 40. Generate a matrix of 20 random values and find its cumulative sum

# %%
a=np.random.rand(20)
np.cumsum(a)


# %%



