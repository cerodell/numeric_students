import numpy as np





# a = np.array([[0.0001, 0.0001, 0.5],[0.5, 1, 1], [0.0001, 1, 0.0001]])

# k = np.linalg.cond(a)

# b = np.array([1, 0.0001, 0.5],[0.5, 1, 1], [0.0001, 1, 0.0001]])


def diffmatrix(n,m):
    x = np.zeros((n,m))
    for i in x:
        B = np.array([(i+1),(-2*i),(i-1)])
    return B

B =diffmatrix(50,40)



# x = range(6)

# for n in x:
#   print(n)


    # n = A.shape[0]
    # E = np.eye(n)
    # if i == j:
    #     E[i,i] = k + 1
    # else:
    #     E[i,j] = k
    # return E @ A