import numpy as np
# from attention import multi_head_attention

def positional_encoding(seq_len, d_model):
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    return pos_encoding

def feed_forward(x, d_ff, d_model):
    W1 = np.random.randn(d_model, d_ff)
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model)
    b2 = np.zeros(d_model)
    
    x = np.dot(x, W1) + b1
    x = np.maximum(0, x)  # ReLU
    x = np.dot(x, W2) + b2
    return x

def encoder_layer(x, num_heads, d_ff, d_model, mask=None):
    # Multi-head attention
    attn_output = multi_head_attention(x, x, x, num_heads, mask)
    
    # Add & Norm
    x = x + attn_output
    x = layer_norm(x)
    
    # Feed forward
    ff_output = feed_forward(x, d_ff, d_model)
    
    # Add & Norm
    x = x + ff_output
    return layer_norm(x)

def decoder_layer(x, enc_output, num_heads, d_ff, d_model, mask=None):
    # Masked multi-head attention (self-attention)
    attn_output = multi_head_attention(x, x, x, num_heads, mask)
    
    # Add & Norm
    x = x + attn_output
    x = layer_norm(x)
    
    # Multi-head attention (encoder-decoder attention)
    attn_output = multi_head_attention(x, enc_output, enc_output, num_heads)
    
    # Add & Norm
    x = x + attn_output
    x = layer_norm(x)
    
    # Feed forward
    ff_output = feed_forward(x, d_ff, d_model)
    
    # Add & Norm
    x = x + ff_output
    return layer_norm(x)

def layer_norm(x, epsilon=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + epsilon)

# Custom tokenizer (simplified for demonstration purposes)
def tokenize(texts, vocab_size=10000, max_len=100):
    # This is a simple character-level tokenizer; trying to improve it later with (BPE, word-level)
    tokenized = []
    for text in texts:
        tokens = np.array([ord(c) % vocab_size for c in text[:max_len]])  # Map characters to integers
        tokens = np.pad(tokens, (0, max_len - len(tokens)), 'constant')  # Pad to max_len
        tokenized.append(tokens)
    return np.array(tokenized)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)

def transformer_encoder(x, num_layers, num_heads, d_ff, d_model, mask=None):
    for _ in range(num_layers):
        x = encoder_layer(x, num_heads, d_ff, d_model, mask)
    return x

def transformer_decoder(x, enc_output, num_layers, num_heads, d_ff, d_model, mask=None):
    for _ in range(num_layers):
        x = decoder_layer(x, enc_output, num_heads, d_ff, d_model, mask)
    return x

def transformer(x, y, num_layers, num_heads, d_ff, d_model, src_mask=None, tgt_mask=None):
    enc_output = transformer_encoder(x, num_layers, num_heads, d_ff, d_model, src_mask)
    dec_output = transformer_decoder(y, enc_output, num_layers, num_heads, d_ff, d_model, tgt_mask)
    return dec_output 

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    
    if mask is not None:
        scores = scores + (mask * -1e9)  # Apply mask (if any)

    attention_weights = softmax(scores)
    output = np.dot(attention_weights, V)
    return output, attention_weights

def multi_head_attention(Q, K, V, num_heads, mask=None):
    d_model = Q.shape[-1]
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    
    depth = d_model // num_heads
    Q_split = np.split(Q, num_heads, axis=-1)
    K_split = np.split(K, num_heads, axis=-1)
    V_split = np.split(V, num_heads, axis=-1)
    
    attention_outputs = []
    for i in range(num_heads):
        attn_output, _ = scaled_dot_product_attention(Q_split[i], K_split[i], V_split[i], mask)
        attention_outputs.append(attn_output)

    concat_attention = np.concatenate(attention_outputs, axis=-1)
    return concat_attention