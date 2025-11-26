import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("experiment_output.pkl", "rb") as f:
    data = pickle.load(f)

ball_pos = np.array([pos[1] for pos in data["ball_pos"]])
quantized = np.floor(ball_pos * 20).astype(int)

unique, counts = np.unique(quantized, return_counts=True)

# Create bar plot
plt.bar(unique, counts, color='skyblue')
plt.xlabel("Number")
plt.ylabel("Frequency")
plt.title("Frequency of Each Number")
plt.xticks(unique)  # ensure each unique number is labeled
plt.show()
