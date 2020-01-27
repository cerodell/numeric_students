import numpy as np

def diffmatrix(n):
    """
    This Function makes....
    """
    A = np.zeros((n,n))
    for index in range(1,n-1):
        A[index,index] = -2
        A[index,index + 1] = A[index,index - 1] = 1
        
    A[0,0] = A[-1,-1] =1

    condtion = np.linalg.cond(A)

    return A, condtion

## Loop and calcualte the condition number for the matrix (B)
for size in range(5,55,5):
    B, condtion = diffmatrix(size)
    print(f"Matrix Size: {B.shape} Condition number: {condtion}")



