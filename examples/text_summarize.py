from os import system,path
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import sys
sys.path.append(path.dirname("../"))
from model.trainer_tpu import train_transformer
from helpers.plot import plot_performance
from model.layers_tpu import tokenize,transformer

def load_and_preprocess_data(dataset_name='cnn_dailymail'):
    # Load dataset
    dataset = load_dataset(dataset_name, '3.0.0')
    
    # Use 'article' as input and 'highlights' as target summaries
    texts = dataset['train']['article'][:1000]  # Limit to 1000 for this example
    summaries = dataset['train']['highlights'][:1000]
    
    # Tokenize both texts and summaries
    X = tokenize(texts)
    Y = tokenize(summaries)
    
    return X, Y

# Split dataset into training and testing sets
def split_data(X, Y, test_size=0.2):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    return X_train, X_test, Y_train, Y_test

# Main execution block
if __name__ == "__main__":
    system("clear")
    # Load and preprocess data
    print("loading data ...")
    X, Y = load_and_preprocess_data()
    print("splittinng ...")
    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=0.2)
    print("training ...")
    # Train the transformer model
    num_layers = 6
    num_heads = 5
    d_ff = 2048
    d_model = 100
    losses, accuracies, test_accuracies = train_transformer(transformer, X_train, Y_train, X_test, Y_test, num_epochs=1)
    print("plotting ...")
    # Plot the results
    plot_performance(losses, accuracies, test_accuracies)
