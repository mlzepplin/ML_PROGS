
import numpy as np
import matplotlib.pyplot as plt

pSuccess = 0.3
numSamples = 100

#init random samples from a uniform disttribution
samples = np.random.uniform(low=0.0, high=1.0, size=numSamples)

#scaled binomial pdf
def binomial(numSuccess,inp):
    result =[]
    p1=np.power(inp, numSuccess)
    o=np.ones((inp.size), dtype=None)
    p2=np.power(o-inp,inp.size-numSuccess)
    result = np.multiply(p1,p2)
    return result

#to get number of successes/heads in samples
def getNumSuccesses(inp,p):
    n=0
    for i in range(0, inp.size):
        if inp[i]<=p:
            n = n+1
    return n

#scaled beta pdf
def beta(a,b,inp):
    result =[]
    p1=np.power(inp, a-1)
    o=np.ones((inp.size), dtype=None)
    p2=np.power(o-inp, b-1)
    result = np.multiply(p1,p2)
    return result



likelihood = binomial(getNumSuccesses(samples,pSuccess),samples)

prior = beta(40,20,samples)

posterior = np.multiply(likelihood,prior)


f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot(samples,likelihood,'*')
ax1.set_title('likelihood')
ax2.plot(samples,prior,'*')
ax2.set_title('prior')
ax3.plot(samples,posterior,'*')
ax3.set_title('posterior')

plt.show()

