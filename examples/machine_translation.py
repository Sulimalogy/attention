from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Adjusted data loading for machine translation
def load_and_preprocess_data(dataset_name='wmt14', src_lang='en', tgt_lang='de', max_len=50, vocab_size=10000):
    # Load the WMT English-to-German dataset
    dataset = load_dataset(dataset_name, 'de-en')
    
    # Extract source (English) and target (German) sentences
    texts = dataset['train']['translation'][src_lang][:1000]  # English source sentences (input)
    translations = dataset['train']['translation'][tgt_lang][:1000]  # German target sentences (output)
    
    # Tokenize both source and target sentences
    X = tokenize(texts, vocab_size=vocab_size, max_len=max_len)  # Tokenized English
    Y = tokenize(translations, vocab_size=vocab_size, max_len=max_len)  # Tokenized German
    
    return X, Y

# Tokenizer (simple character-based for demonstration, can replace with better tokenizers)
def tokenize(texts, vocab_size=10000, max_len=50):
    tokenized = []
    for text in texts:
        tokens = np.array([ord(c) % vocab_size for c in text[:max_len]])  # Convert characters to tokens
        tokens = np.pad(tokens, (0, max_len - len(tokens)), 'constant')  # Pad to max_len
        tokenized.append(tokens)
    return np.array(tokenized)

# Example of training the Transformer model for machine translation

# Main execution block
if __name__ == "__main__":
    system("clear")
    # Load and preprocess data for translation task (English to German)
    print("loading data ...")
    X, Y = load_and_preprocess_data(dataset_name='wmt14', src_lang='en', tgt_lang='de', max_len=50, vocab_size=10000)
    print("splitting ...")
    
    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=0.2)
    
    print("training ...")
    # Train the transformer model
    num_layers = 6
    num_heads = 8
    d_ff = 2048
    d_model = 512  # Model dimensionality (adjust if needed)
    
    losses, accuracies, test_accuracies = train_transformer(transformer, X_train, Y_train, X_test, Y_test, num_epochs=10)
    
    print("plotting ...")
    # Plot the results
    plot_performance(losses, accuracies, test_accuracies)
