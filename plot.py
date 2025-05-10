import matplotlib.pyplot as plt

# Sample data (replace with your lists)
list1 = [1282.18, 962.98, 1208.18, 1141.58, 883.32, 786.84, 1052.02, 1166.52, 999.04, 970.92, 1325.86, 1381.92, 1258.64, 980.92, 1167.92, 785.9, 953.78, 1248.58, 1091.64, 699.46, 1249.64, 1519.56, 1039.14, 1591.8, 934.4, 1293.12, 633.78, 1451.08, 1034.66, 771.28]
list2 = [1, 3, 5, 7, 9, 11, 13]
list3 = [5, 4, 3, 2, 1, 0, -1, -2, -3]

# Create figure and axis
plt.figure(figsize=(10, 6))

# Plot each list with different styles
plt.plot(range(len(list1)), list1, label='List 1', markersize=8, linewidth=2)
plt.plot(range(len(list2)), list2, label='List 2', markersize=8, linewidth=2)
plt.plot(range(len(list3)), list3, label='List 3', markersize=8, linewidth=2)

# Add labels and title
plt.xlabel('Index Position', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('Visualization of Three Lists', fontsize=14, fontweight='bold')

# Add legend and grid
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout and show plot
plt.tight_layout()
plt.show()