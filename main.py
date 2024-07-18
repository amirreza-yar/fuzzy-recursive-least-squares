import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def rls_algo(n_features, data):
    n_features = 4 # Number of inputs
    w = np.zeros(n_features) # The theta matrix
    P = np.eye(n_features) * 1000  # Large initial covariance, alpha = 1000
    lambda_ = 1  # Forgetting factor, Default 1

    y_pred = np.zeros(len(data)) # Initialize array for predictions values
    J_p = np.zeros(len(data)) # Initialize array for cost function values

    for p in range(len(data)):
        x_p = data.iloc[p, :-1].values # The value of "b"
        y_p = data.iloc[p, -1]
    
        y_pred[p] = np.dot(x_p, w)
        error = y_p - y_pred[p]
           
        K_p = np.dot(P, x_p) / (lambda_ + np.dot(x_p.T, np.dot(P, x_p)))
        w += K_p * error # The "theta"
        P = (P - np.outer(K_p, np.dot(x_p.T, P))) / lambda_
           
        # Update the cost function J_t(w) incrementally
        if p == 0:
            J_p[p] = error**2
        else:
            J_p[p] = lambda_ * J_p[p-1] + error**2

    return w, P, y_pred, J_p

# Import the data from the csv file
data = pd.read_csv('data/rls_data_unsortted.csv')

# Compute the "Y_prediction" and the "theta" matrixs
w, P, y_pred, J_p = rls_algo(4, data)

print(f"data shape: {data.shape}", f"Theta matrix: {w}", sep='\n')

plt.figure(figsize=(14, 6))
   
plt.subplot(1, 3, 1)
plt.plot(data.index, data['y'], label='True Values')
plt.plot(data.index, y_pred, label='Predicted Values', linestyle='--')
plt.xlabel('Index')
plt.ylabel('y')
plt.legend()
plt.title('True vs Predicted Values')
plt.grid()

plt.subplot(1, 3, 2)
plt.plot(range(len(data)), J_p, label='Cost Function J_p(w)')
plt.xlabel('Index')
plt.ylabel('J_p(w)')
plt.legend()
plt.title('Cost Function J_p(w)')
plt.grid()

plt.subplot(1, 3, 3)
plt.plot(range(len(data)), data['y'] - y_pred, label='Error')
plt.xlabel('Index')
plt.ylabel('Y - Y_pred')
plt.legend()
plt.title('Error of the model')
plt.grid()

plt.tight_layout()
plt.show()

# Import the data from the csv file
data = pd.read_csv('data/rls_data_unsortted.csv')
train_data = data[:60]
test_data = data[60:]

# Compute the "Y_prediction" and the "theta" matrixs
w, P, y_pred, J_p = rls_algo(4, train_data)

print(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}", f"Theta matrix: {w}", sep='\n')

# Computing the y_pred for the test data
test_data = test_data[['x1', 'x2', 'x3', 'x4', 'y']].to_numpy()
test_xs = np.array([test_data[:, 0], test_data[:, 1], test_data[:, 2], test_data[:, 3],])
test_ys = np.array([test_data[:, 4]])[0]
test_y_pred = w @ test_xs

plt.figure(figsize=(14, 6))
   
plt.subplot(1, 2, 1)
plt.plot(range(len(test_ys)), test_ys, label='True Values')
plt.plot(range(len(test_y_pred)), test_y_pred, label='Predicted Values', linestyle='--')
plt.xlabel('Index')
plt.ylabel('y')
plt.legend()
plt.title('TEST True vs Predicted Values')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(range(len(test_ys)), test_ys - test_y_pred, label='TEST Error')
plt.xlabel('Index')
plt.ylabel('TEST_Y - TEST_Y_pred')
plt.legend()
plt.title('Error of the TEST')
plt.grid()

plt.tight_layout()
plt.show()