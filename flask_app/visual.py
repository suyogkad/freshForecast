import matplotlib.pyplot as plt

# Values
metrics = [1.1897418882329553, 12.403789467756705, 0.997957962951381]
metric_labels = ['MAE', 'MSE', 'R^2 Score']
colors = ['blue', 'red', 'green']

# Basic Bar Chart
plt.figure(figsize=(10,6))
plt.bar(metric_labels, metrics, color=colors, alpha=0.7)

# Adding Text Labels
for i in range(len(metrics)):
    plt.text(i, metrics[i] + 0.05, round(metrics[i], 4), ha='center')

# Title and Labels
plt.title('Model Evaluation Metrics')
plt.ylabel('Value')
plt.ylim(0, 13)

# Display Plot
plt.show()
