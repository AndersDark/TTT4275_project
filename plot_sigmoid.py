import numpy as np
import matplotlib.pyplot as plt

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Create range of input values
z = np.linspace(-10, 10, 1000)
s = sigmoid(z)

# Plot
plt.figure(figsize=(6, 4))
plt.plot(z, s, label="Sigmoid", color='blue')
plt.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("sigmoid(z)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
