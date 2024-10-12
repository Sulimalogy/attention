# Transformer model with Numpy  

This repository contains a simple implementation of a **Transformer** model using **Numpy** only. The project avoids using deep learning frameworks like TensorFlow or PyTorch and relies on basic matrix operations, making it ideal for educational purposes and understanding how the Transformer architecture works.

## Features
- Implements core Transformer components (Multi-Head Attention, Positional Encoding, Feed-Forward Network, etc.) from scratch using Numpy.
- 3 Examples on using it like Text Summerization, Text Generation, Machine Trnssilation.
- Includes training and evaluation loops to compute losses and accuracy.
- Visualizes model performance (loss and accuracy) over time using Matplotlib.
- Can be extended to work on Google Colab with TPU/GPU using Numpy.

## Key Components
- **Scaled Dot Product Attention**: The core of attention mechanism, used to compute attention weights.
- **Multi-Head Attention**: Splits the input into multiple attention heads, processes them in parallel, and concatenates the results.
- **Positional Encoding**: Adds positional information to the input embeddings so that the model can take the order of words into account.
- **Feed-Forward Network**: A simple fully connected layer with ReLU activation used after attention in each Transformer block.
- **Layer Normalization**: A normalization technique applied after the attention and feed-forward steps to stabilize training.

## Files
- `transformer.py`: Contains the full implementation of the Transformer model and its components using Numpy.
- `data_preprocessing.py`: Functions to load and preprocess the dataset, including tokenization and train-test splitting.
- `train.py`: Training script that trains the Transformer model on the dataset, computes the loss and accuracy, and evaluates the model on the test set.
- `plot_performance.py`: A script to plot training and testing loss/accuracy over time.

## Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/Sulimalogy/attention.git
cd attention
pip install -r requirements.txt
```
## Running Examples
```bash
python3 -m examples.text_summarize 
# or
python3 -m examples.text_generation
# or 
python3 -m examples.machine_translation
```

## Performance Evaluation
The performance of the model (training loss, training accuracy, and test accuracy) is plotted using Matplotlib at the end of training.

## Future Enhancements
- GPU/TPU support using Numpy or low-level libraries.
- Optimization of matrix operations for faster training.
- More complex tokenization methods such as byte-pair encoding (BPE).
- Adding support for other NLP tasks like translation, question answering, etc.

## License
This project is licensed under the Apache License. See the [LICENSE](LICENSE) file for details.

## Reference
- Original paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Dataset: CNN/Daily Mail summarization dataset via Hugging Face Datasets.