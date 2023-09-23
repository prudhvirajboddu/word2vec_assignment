### YOUR CODE HERE

### Please use your implementation of sigmoid in here.
'''
centerWordVec: 1 * d
outsideVectors: n * d
'''
#1 * m
vec = centerWordVec.dot(outsideVectors[indices].T)
vec[1:] *= -1
sig = sigmoid(vec)
tmp = np.log(sig)
loss = -tmp[0] - np.sum(tmp[1:])
#1 * m
t1 = 1 - sig
gradCenterVec = t1.dot(outsideVectors[indices]) - 2 * t1[0] * outsideVectors[outsideWordIdx]
#累加
gradOutsideVecs = np.zeros_like(outsideVectors)
gradOutsideVecs[outsideWordIdx] += -t1[0] * centerWordVec
for i in range(K):
    k = negSampleWordIndices[i]
    gradOutsideVecs[k] += t1[i + 1] * centerWordVec
### END YOUR CODE