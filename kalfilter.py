import numpy as np

class KF:
    def __init__(self, F, H, Q, R, G, P, x0):

        self.n = F.shape[1]

        #system matricies
        self.F = F
        self.H = H
        self.G = G
        
        self.Q = Q
        self.P = P
        self.R = R

        self.x = x0 # init conditions
    
    # predict with a control variable passed 
    # through prediction matrix G
    def predict(self, u):
        # see textbook for algorithm
        self.x = np.dot(self.F, self.x) + np.dot(self.G, u)

        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x
    
    # update with measurement
    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

def example():
    # STDev on control variable (drifty, small noise)
    sigma_a = 2.0 

    # STDev on global measurement (objective, high noise)
    sigma_z = 20.0
    
    dt = 0.25
    Nsteps = 100

    F = np.array([[1, dt], [0, 1]])
    H = np.array([1, 0]).reshape(1, 2)

    G = np.array([1.0/2 * dt**2, dt]).reshape(2,1)

    Q = G @ G.T * (sigma_a*sigma_a)
    R = np.array([sigma_z**2]).reshape(1, 1)

    x = np.arange(Nsteps) * dt
    #real acceleration
    zs = 0.5 * 30 * x ** 2 + np.random.normal(0, sigma_z**2, len(x))

    print(len(zs))

    us = np.random.normal(30, sigma_a**2, len(zs))
    # measurements = - (x**2 + 2*x - 2)  + np.random.normal(0, sigma_z**2, len(zs))
    kf = KF(F = F, H = H, Q = Q, R = R, G=G, x0=np.array([0,0]).reshape(2,1), P=np.diag([500] * 2))
    predictions = []

    for u, z in zip(us, zs):
        predictions.append(float(np.dot(H, kf.predict(u=u))[0][0]))
        kf.update(z)

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax2.plot(range(len(zs)), zs, label = 'Measurements')
    ax2.plot(range(len(predictions)), np.array(predictions), label = 'Kalman Filter Prediction')
    ax2.plot(range(len(zs)), 0.5 * 30 * x ** 2, label = "Truth")

    truth = 0.5 * 30 * x ** 2
    print(truth)
    print(truth - predictions)

    ax1.plot(range(len(zs)), (truth - zs), label = 'Measurement Error')
    ax1.plot(range(len(predictions)), (truth - predictions), label = 'Prediction Error')
    ax1.legend()
    plt.legend()
    plt.show()


def example2():
    # STDev on control variable (drifty, small noise)
    sigma_a = 2.0 

    # STDev on global measurement (objective, high noise)
    sigma_z = 20.0
    
    dt = 0.25
    Nsteps = 200

    F = np.array([[1, dt], [0, 1]])
    H = np.array([1, 0]).reshape(1, 2)

    G = np.array([1.0/2 * dt**2, dt]).reshape(2,1)

    Q = G @ G.T * (sigma_a*sigma_a)
    R = np.array([sigma_z**2]).reshape(1, 1)

    x = np.arange(Nsteps) * dt
    #real acceleration
    zs = 500*np.sin(x / 5) + np.random.normal(0, sigma_z**2, len(x))

    print(len(zs))

    # us = np.random.normal(30, sigma_a**2, len(zs))
    us = np.zeros(len(zs))
    for i in range(0,len(zs)):
        us[i] = -500.0/25 * np.sin(x[i]) + np.random.normal(0, sigma_a**2)
    print(us)
    # measurements = - (x**2 + 2*x - 2)  + np.random.normal(0, sigma_z**2, len(zs))
    kf = KF(F = F, H = H, Q = Q, R = R, G=G, x0=np.array([0,0]).reshape(2,1), P=np.diag([500] * 2))
    predictions = []

    for u, z in zip(us, zs):
        predictions.append(float(np.dot(H, kf.predict(u=u))[0][0]))
        kf.update(z)

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    truth = 500*np.sin(x / 5)
    ax2.plot(range(len(zs)), zs, label = 'Measurements')
    ax2.plot(range(len(predictions)), np.array(predictions), label = 'Kalman Filter Prediction')
    ax2.plot(range(len(zs)), truth, label = "Truth")

    
    print(truth)
    print(truth - predictions)

    ax1.plot(range(len(zs)), (truth - zs), label = 'Measurement Error')
    ax1.plot(range(len(predictions)), (truth - predictions), label = 'Prediction Error')
    ax1.legend()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    example2()