
    
    Today we will discuss the Numpy library
    
import numpy as np
np.__version__

The basic data structure in numpy is an n-dimensional array.

a = np.array([1, 2, 3, 4, 5])
b = np.array([1.1, 2.2, 3.3, 4.4, 5.5])

c = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype = 'float32')
d = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype = 'int32')

e = np.array([1])
f = np.array([[1]])
g = np.array([[[1]]])

e.ndim
f.ndim
g.ndim

h = np.array([range(i, i+3) for i in [2, 4, 6]])

i = np.zeros(10, dtype = int)
np.zeros((2, 5))

np.ones(10)
j = np.ones((3, 5), dtype = int)

k = np.full((3, 5), 3.14)

range ----> arange

np.arange(0, 10)
np.arange(0, 20, 3)
l = np.arange(0, 20)
m = np.linspace(0, 20)

np.linspace(0, 10, 5)

n = np.random.randn(4, 4)
o = np.random.rand(4, 4)
p = np.random.randint(4)
q = np.random.randint(0, 10, (4, 4))

r = np.eye(4)

s = np.empty((3, 3))

The common operations on numpy arrays are:
    
    1. Determining attributes of array
    2. Indexing
    3. Slicing & Dicing
    4. Reshaping
    5. Joining and Splitting
    
x1 = np.random.randint(10, size = 6)
x2 = np.random.randint(10, size = (3, 4))
x3 = np.random.randint(10, size = (3, 4, 5))

x3.ndim
x3.shape
x3.size
x3.dtype
x3.itemsize
x3.nbytes

x1
x1[0]
x1[4]
x1[5]
x1[-1]

x2
x2[0]
x2[0, 0]
x2[-1, -1]

x2[0, 0] = 12
x2[0, 0] = 12.34

x = np.arange(10)
x

x[0:5]
x[:5]

x[5:]
x[4:7]

x[:]
x[::2]
x[::3]
x[::-1]
x[::-3]

x2

x2[rows, cols]

x2[:2, :3]
x2[1, :]
x2[:, 3]

x2[2, 1]

Note : When we invoke the slice of an array, it returns a 
view and not a copy.

print(x2)

x2_sub = x2[:2, :3]
x2_sub[0, 0] = 101

print(x2)

x2_s = x2[:2, :2]
x2_s[:] = 100

print(x2)

x2_sub_copy = x2[:2, :2].copy()
x2_sub_copy[:] = 7

grid = np.arange(1, 10)
grid = grid.reshape((3, 3))

x = np.array([1, 2, 3])
y = np.array([3, 2, 1])

np.concatenate([x, y])

z = np.array([99, 99, 99])

np.concatenate([x, y, z])

grid = np.array([[1, 2, 3],
                 [4, 5, 6]])

grid
np.concatenate([grid, grid])
np.concatenate([grid, grid], axis = 1)

x = [1, 4, 5, 99, 99, 4, 8, 7]

x1, x2, x3 = np.split(x, [3, 5])
x1
x2
x3

my_list = list(range(1000000))
my_arr = np.array(range(1000000))

%time for i in range(10): my_list2 = my_list * 2
%time for i in range(10): my_arr2 = my_arr * 2

[1, 2, 3] * 3
np.array([1, 2, 3]) * 3

mat = np.array([[1, 2], [3, 4]])

mat * mat
mat @ mat
































































































