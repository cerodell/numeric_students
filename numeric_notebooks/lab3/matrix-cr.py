import numpy as np





a = np.array([[0.0001, 0.0001, 0.5],[0.5, 1, 1], [0.0001, 1, 0.0001]])

k = np.linalg.cond(a)