import numpy as np

def compute_accuracy(y_true, y_pred):
    if len(y_true.shape) == 1:
        y_true = np.expand_dims(y_true, axis=0)
    if len(y_pred.shape) == 1:
        y_pred = np.expand_dims(y_pred, axis=0)

    correct_predictions = np.sum(np.argmax(y_true, axis=-1) == np.argmax(y_pred, axis=-1))
    total_predictions = y_true.shape[0] * y_true.shape[1]
    return correct_predictions / total_predictions

# Train the transformer model
def train_transformer(model, X_train, Y_train, X_test, Y_test, num_epochs=10, lr=0.001):
    losses = []
    accuracies = []
    test_accuracies = []
    num_layers = 6 
    num_heads = 5
    d_ff = 2048
    d_model = 100
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        for i in range(len(X_train)):
            x = X_train[i]
            y = Y_train[i]
            y_pred = model(x, y,num_layers,num_heads,d_ff,d_model)
            
            # Loss: Mean Squared Error (simplified)
            loss = np.mean((y - y_pred) ** 2)
            epoch_loss += loss
            
            # Compute accuracy
            accuracy = compute_accuracy(y, y_pred)
            epoch_accuracy += accuracy
            print(f"\rEpoch: {epoch+1}/{num_epochs} ,Step: {i+1}/{len(X_train)}", end='')  # Print the count, overwrite with \r
        losses.append(epoch_loss / len(X_train))
        accuracies.append(epoch_accuracy / len(X_train))
        
        # Evaluate on test set
        test_accuracy = np.mean([compute_accuracy(Y_test[j], model(X_test[j], Y_test[j],num_layers,num_heads,d_ff,d_model)) for j in range(len(X_test))])
        test_accuracies.append(test_accuracy)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(X_train)}, Train Accuracy: {epoch_accuracy / len(X_train)}, Test Accuracy: {test_accuracy}')
    
    return losses, accuracies, test_accuracies