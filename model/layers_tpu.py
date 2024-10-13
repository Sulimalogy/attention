import numpy as np
import tensorflow.experimental.numpy as tfnp
# from attention import multi_head_attention
import tensorflow as tf

def positional_encoding(seq_len, d_model):
    with tf.device("TPU:0"):
        position = tfnp.arange(seq_len)[:, tfnp.newaxis]
        div_term = tfnp.exp(tfnp.arange(0, d_model, 2) * -(tfnp.log(10000.0) / d_model))
        pos_encoding = tfnp.zeros((seq_len, d_model))
        pos_encoding[:, 0::2] = tfnp.sin(position * div_term)
        pos_encoding[:, 1::2] = tfnp.cos(position * div_term)
    return pos_encoding

def feed_forward(x, d_ff, d_model):
    with tf.device("TPU:0"):
        W1 = tfnp.random.randn(d_model, d_ff)
        b1 = tfnp.zeros(d_ff)
        W2 = tfnp.random.randn(d_ff, d_model)
        b2 = tfnp.zeros(d_model)
        
        x = tfnp.dot(x, W1) + b1
        x = tfnp.maximum(0, x)  # ReLU
        x = tfnp.dot(x, W2) + b2
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
    with tf.device("TPU:0"):
        mean = tfnp.mean(x, axis=-1, keepdims=True)
        std = tfnp.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + epsilon)

# Custom tokenizer (simplified for demonstration purposes)
def tokenize(texts, vocab_size=10000, max_len=100):
    # This is a simple character-level tokenizer; trying to improve it later with (BPE, word-level)
    tokenized = []
    with tf.device("TPU:0"):
        for text in texts:
            tokens = tfnp.array([ord(c) % vocab_size for c in text[:max_len]])  # Map characters to integers
            tokens = tfnp.pad(tokens, (0, max_len - len(tokens)), 'constant')  # Pad to max_len
            tokenized.append(tokens)
    return tfnp.array(tokenized)

def softmax(x):
    with tf.device("TPU:0"):
        exps = tfnp.exp(x - tfnp.max(x, axis=-1, keepdims=True))
    return exps / tfnp.sum(exps, axis=-1, keepdims=True)

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
    with tf.device("TPU:0"):
        scores = tfnp.dot(Q, K.T) / tfnp.sqrt(d_k)
    
        if mask is not None:
            scores = scores + (mask * -1e9)  # Apply mask (if any)

        attention_weights = softmax(scores)
        output = tfnp.dot(attention_weights, V)
    return output, attention_weights

def multi_head_attention(Q, K, V, num_heads, mask=None):
    d_model = Q.shape[-1]
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    with tf.device("TPU:0"):
        depth = d_model // num_heads
        Q_split = tfnp.split(Q, num_heads, axis=-1)
        K_split = tfnp.split(K, num_heads, axis=-1)
        V_split = tfnp.split(V, num_heads, axis=-1)
        
        attention_outputs = []
        for i in range(num_heads):
            attn_output, _ = scaled_dot_product_attention(Q_split[i], K_split[i], V_split[i], mask)
            attention_outputs.append(attn_output)

        concat_attention = tfnp.concatenate(attention_outputs, axis=-1)
    return concat_attention