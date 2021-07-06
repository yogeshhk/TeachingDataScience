# https://www.youtube.com/watch?v=E-I2DNVzQLg&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=5

# Step 1
#   - Prediction : Manual
#   - Gradient Computation : Manual
#   - Loss Computation : Manual
#   - Parameter Update : Manual


import numpy as np
X = np.array([1,2,3,4],dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)

w = 0.0 # some seed start

# Prediction
def forward(x):
    return w * x

def loss(y,y_predicted):
    return ((y_predicted - y)**2).mean()

# gradient
# loss = 1/N (wx - y)**2
# dloss/dw = 1/N * 2(wx-y).x
def gradient(x,y,y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

# prediction before training for x = 5
print(f"prediction before training : {forward(5)}")

# Training
learning_rate = 0.01
n_iters = 10
for epoch in range(n_iters):
    # prediction
    y_pred = forward(X)
    l = loss(Y, y_pred)
    dw = gradient(X,Y,y_pred)
    w -= learning_rate * dw
    print(f"epoch {epoch+1}, w = {w}, loss = {l}")

print(f"prediction after training: {forward(5)}")

