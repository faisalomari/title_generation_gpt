import matplotlib.pyplot as plt

# Read the training log file and extract data
with open('traininglog.txt', 'r') as file:
    lines = file.readlines()

epochs = []
grad_norms = []
learning_rates = []

for line in lines:
    data = eval(line)
    epochs.append(data['epoch'])
    grad_norms.append(data['grad_norm'])
    learning_rates.append(data['learning_rate'])

# Plotting
plt.figure(figsize=(10, 6))

# Plot grad_norm
plt.subplot(2, 1, 1)
plt.plot(epochs, grad_norms, marker='o', linestyle='-')
plt.title('Grad Norm Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Grad Norm')

# Plot learning_rate
plt.subplot(2, 1, 2)
plt.plot(epochs, learning_rates, marker='o', linestyle='-')
plt.title('Learning Rate Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')

plt.tight_layout()
plt.savefig("graph.png")
plt.show()
