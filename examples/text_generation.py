from datasets import load_dataset

# Data loading for text generation (using PTB dataset or any text corpus)
def load_and_preprocess_data(dataset_name='ptb_text_only', max_len=100, vocab_size=10000):
    # Load dataset (Penn Treebank or similar)
    dataset = load_dataset(dataset_name)
    
    # Extract the 'text' field (assuming dataset has a text field, adjust if different)
    texts = dataset['train']['text'][:1000]  # Limit to 1000 for this example
    
    # Tokenize the texts for character-based generation
    X, Y = prepare_for_text_generation(texts, max_len=max_len, vocab_size=vocab_size)
    
    return X, Y

# Prepare input-output pairs for text generation
def prepare_for_text_generation(texts, max_len=100, vocab_size=10000):
    tokenized_input = []
    tokenized_output = []
    
    for text in texts:
        for i in range(len(text) - max_len):
            input_seq = text[i:i+max_len]
            output_seq = text[i+1:i+1+max_len]
            
            input_tokens = [ord(c) % vocab_size for c in input_seq]
            output_tokens = [ord(c) % vocab_size for c in output_seq]
            
            tokenized_input.append(input_tokens)
            tokenized_output.append(output_tokens)
    
    return np.array(tokenized_input), np.array(tokenized_output)

# Tokenizer (simple character-based tokenizer)
def tokenize(texts, vocab_size=10000, max_len=100):
    tokenized = []
    for text in texts:
        tokens = np.array([ord(c) % vocab_size for c in text[:max_len]])
        tokens = np.pad(tokens, (0, max_len - len(tokens)), 'constant')
        tokenized.append(tokens)
    return np.array(tokenized)

# Example of training the Transformer model for text generation

# Main execution block
if __name__ == "__main__":
    system("clear")
    
    # Load and preprocess data for text generation (Penn Treebank or similar corpus)
    print("loading data ...")
    X, Y = load_and_preprocess_data(dataset_name='ptb_text_only', max_len=100, vocab_size=10000)
    
    print("splitting ...")
    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=0.2)
    
    print("training ...")
    # Train the transformer model
    num_layers = 6
    num_heads = 8
    d_ff = 2048
    d_model = 256  # Adjust model dimensionality for text generation task
    
    losses, accuracies, test_accuracies = train_transformer(transformer, X_train, Y_train, X_test, Y_test, num_epochs=10)
    
    print("plotting ...")
    # Plot the results
    plot_performance(losses, accuracies, test_accuracies)
