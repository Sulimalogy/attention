import matplotlib.pyplot as plt

def plot_performance(losses, accuracies,test_accuracies):
    epochs = range(1, len(losses) + 1)
    
    plt.figure(figsize=(18, 6))  # Adjusted figure size

    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, losses, label='Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    # Plot Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, accuracies, label='Accuracy', color='green')
    plt.title('Training Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    
    # Plot Testing Accuracy
    plt.subplot(1, 3, 3)
    plt.plot(epochs, test_accuracies, label='Test Accuracy', color='blue')
    plt.title('Testing Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.show()