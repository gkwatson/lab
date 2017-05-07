import numpy as np

arr = np.array([[[1,2,3,4],[1,2,3,4],[1,2,3,4]],
[[1,2,3,4],[1,2,3,4],[1,2,3,4]],
[[1,2,3,4],[1,2,3,4],[1,2,3,4]]])

print(arr)
print
print
print(arr[:,:,:3])
print
print
print(arr[:,:,3:].reshape(-1))
