import numpy as np 


def llfr(A, R, f, q1):
	"""
	Args:
	param A (numpy.array): The inter-item Correlation Matrix, shape =(m,m).
	param R (numpy.array): The Rating Matrix, shape=(n,m).
	param f (int): the number of latent factors.
	param q1 (numpy.array): A random unit vector, shape=(m).

	Returns:
	numpy.array : Matrix Pi whose rows are the recommendation vectors for every user, shape=(n,m).
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

