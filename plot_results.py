import matplotlib.pyplot as plt

# Sample data (replace with your actual experiment results)
experiment_1 = [79.5351, 79.7601, 79.3631, 78.9702, 78.9188, 77.9993, 78.4221, 77.5501]
experiment_2 = [78.5187, 76.5729, 75.1886, 74.4275, 73.5399, 72.4902, 71.7426, 71.1045]
experiment_3 = [76.0364, 72.9047, 69.1346, 66.7937, 64.629, 62.4287, 60.6374, 59.3307]

# Create figure and axis
plt.figure(figsize=(10, 6))

# Plot each experiment with distinct styles
plt.plot(range(1,9),experiment_1, 

         color='blue', 
         label='Ev. Fraca')
plt.plot(range(1,9),experiment_2, 

         color='red', 
         label='Ev. Media')
plt.plot(range(1,9),experiment_3, 

         color='green', 
         label='Ev. Forte')

# Add labels and title
plt.xlabel('Quantidade de evidencias', fontsize=12)
plt.ylabel('Tempo', fontsize=12)
#plt.title('Comparison of Three Experiments', fontsize=14)

# Add legend and grid
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Customize tick parameters
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

# Add tight layout to optimize spacing
plt.tight_layout()

# Display the plot
plt.show()