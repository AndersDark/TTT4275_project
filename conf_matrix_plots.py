import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

conf_matrix = np.array([[20, 0, 0],
                        [0, 20, 0],
                        [0, 1, 19]])



labels = ["Setosa", "Versicolor", "Virginica"]

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)

plt.xlabel("Predicted Class", fontsize = 12)
plt.ylabel("True Class", fontsize = 12)
plt.title("Confusion Matrix - Error Rate: 1.69%")
plt.tight_layout()
plt.show()