import numpy as np

def create_random_transition_matrix(n):
    #random values [0,1]
    P = np.random.rand(n, n)
    #normalize each row sum to 1
    P /= P.sum(axis=1, keepdims=True)
    return P

def create_random_distribution(n):
    p = np.random.rand(n)
    #normalize sum(p) = 1
    p /= p.sum()
    return p

def apply_transition(P, p, steps=50):
    #transition rule
    PT = P.T
    for _ in range(steps):
        p = np.dot(PT, p)
    return p

def compute_stationary_distribution(P):
    eigvals, eigvecs = np.linalg.eig(P.T)
    #eigenvalue closest to 1
    idx = np.argmin(np.abs(eigvals - 1))
    v = np.real(eigvecs[:, idx])
    #rid of negative components
    v = np.abs(v)
    #normalize
    v /= v.sum()
    return v

def main():
    #set by assignment
    n = 5

    #random 5x5 matrix
    P = create_random_transition_matrix(n)
    print("Transition matrix (P):\n", P, "\n")

    #probability vector
    p = create_random_distribution(n)
    print("Initial probability vector (p):\n", p, "\n")

    #transition rule done 50 times
    p50 = apply_transition(P, p, steps=50)
    print("Probability vector after 50 transitions:\n", p50, "\n")

    #stationary distribution
    stationary = compute_stationary_distribution(P)
    print("Stationary distribution (v):\n", stationary, "\n")

    #compute difference
    diff = np.abs(p50 - stationary)
    print("Component-wise absolute difference:\n", diff, "\n")
    print("Maximum difference:", np.max(diff))

    #check their error
    if np.allclose(p50, stationary, atol=1e-5):
        print("\np50 and stationary distribution match within 10^-5.")
    else:
        print("\np50 and stationary distribution do not match within 10^-5.")

#run above code
if __name__ == "__main__":
    main()