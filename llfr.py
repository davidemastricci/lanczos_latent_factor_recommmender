import numpy as np 


def llfr(A, R, f, q1):
	"""
	param A: The inter-item Correlation Matrix, numpy.array shape =(m,m).
	param R: The Rating Matrix, numpy.array shape=(n,m).
	param f: the number of latent factors.
	param q1: A random unit vector, numpy.array shape=(m).
	return : Matrix Pi whose rows are the recommendation vectors for every user, numpy.array shape=(n,m).
	"""
    m = A.shape[0]
    Q = np.ones(shape=(m,f+2))
    Q[:,0]=np.zeros(m)
    Q[:,1]= q1
    b = 0
    for i in range(1, f+1):
        w = np.dot(A, Q[:,[i]]) - b*Q[:,[i-1]]
        a = np.dot(w.T,Q[:,[i]])[0][0]
        w = w - a*Q[:,[i-1]]
        b = np.linalg.norm(w, ord=2)
        Q[:,i+1] = (w / b).T[0]
    Q = np.delete(Q,[0,1],1)
    Pi=np.dot(np.dot(R,Q),Q.T)
    return Pi

