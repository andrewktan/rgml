import matplotlib.pyplot as plt
import numpy as np

# parameters
numsamples = 20000

# sim parameters
sz = 81
beta = 0.45  # inv temperature
J = 1  # nearest neighbor coupling
h = 0  # 0.1  # external field

Z = sz**3  # iterations

# sample generating loop

for ns in range(numsamples):
    # initialize random field
    #    beta = 1 / (np.random.rand() * 300)

    field = np.random.randint(2, size=(sz, sz)) * 2 - 1

    for i in range(Z):
        r, c = np.random.randint(sz, size=2)

        # calculate energy difference
        s = 0
        s += field[np.mod(r+1, sz), np.mod(c, sz)]
        s += field[np.mod(r-1, sz), np.mod(c, sz)]
        s += field[np.mod(r, sz), np.mod(c+1, sz)]
        s += field[np.mod(r, sz), np.mod(c-1, sz)]
        s *= field[r, c]

        pflip = 1 / np.exp(2*beta*(J*s + h*field[r, c]))

        if np.random.rand() < pflip:
            field[r, c] *= -1

#    plt.matshow(field, cmap=plt.cm.gray)
#    plt.show()

    sample = field.reshape((1, -1))
    label = np.array([beta])

    with open('data_0_45', 'ab') as datafile:
        np.savetxt(datafile, sample.astype(int), fmt='%i')

#    with open('labels','ab') as labelfile:
#        np.savetxt(labelfile, label)

    print(ns, '%.4f' % beta)
