# Language Models Implementation Repository

## Project Overview

This repository is mainly created for educational purposes (mainly my own), with an emphasis on the practical implementation of state-of-the-art (SOTA) language models papers, utilizing the PyTorch library.

The main goal of this project is to provide a comprehensive and detailed implementation of the most recent and popular language models, such as GPT-2, Llama 2, Mistral, and others, as well as to provide a detailed explanation of the underlying concepts and mechanisms of these models.

## Results
- GPT:
  - Context: First training on Lightning.ai with T4 GPU. Small models, no expectation of good results just to test the code.
  - Dataset: The spanish bible. (bible_es.txt)
  - Params: 2 layers, 2 heads, 384 embedding size, 50304 vocab size (gpt2 tokenizer), 6e-5 learning rate, 256 block size, 64 batch size.
  - Results: Steps: 4600, training loss: 2.1826, validation loss: 3.1964
- GPT+RoPE:
  - Context: Same model but without the positional embedding. I implemented the RoPE (Relative Positional Encoding) instead.
  - Dataset: The spanish bible. (bible_es.txt)
  - Params: 2 layers, 2 heads, 384 embedding size, 50304 vocab size (gpt2 tokenizer), 6e-5 learning rate, 256 block size, 64 batch size. Total: 42.31M params.
  - Results: Steps: 4600, training loss: 2.1557, validation loss: 3.1664
  - Conclusions: The model with RoPE is slightly better than the model with positional embeddings. The model with RoPE has a lower training loss and validation loss.

  
## Future Work and TODO's

The following are among the planned future works and 'To Do' items for this project:

### Model/Architecture improvements:

- [x] GPT-2
- [x] Implement GeLU instead of RELU
- [ ] Combine the `Head` and `MultiHeadAttention` into one class that processes all the heads in parallel, treating the heads as another batch dimension.
- [x] Take a look to Flash Attention (https://arxiv.org/pdf/2205.14135.pdf)
- [x] Implement RoPE
- [ ] Implement Mixture of Experts (Mixtral)
- [ ] Implement Grouped Query Attention (GQA, Llama2-3)
- [ ] Mistral 7B
- [ ] Mixtral 8x7B
- [ ] Llama 2 8B
- [ ] Mamba
- [ ] Jamba
- [ ] Implement the Transformer-XL
- [ ] Implement linear transformer
- [ ] Implement Infinite attention (https://arxiv.org/pdf/2404.07143.pdf)
- [ ] Study if Infinite attention can be implemented on top of pre-trained models like Mixtral of Experts
- [ ] BitNet: Scaling 1-bit Transformers for Large Language Models (https://arxiv.org/pdf/2310.11453.pdf)

### Fine-tuning improvements:

- [ ] Load pre-trained models
- [ ] Implement LoRA
- [ ] Implement QLoRA

### Training improvements:

- [ ] New SOTA AdamW optimizer
- [ ] Take a look at chinchilla (https://arxiv.org/pdf/2205.14135.pdf)
- [ ] Implement flash attention to speed up training
- [ ] Implement caching for the attention mechanism (across the model)
- [ ] Dynamic learning rate
- [ ] Implement checkpoints
- [ ] Better visualization of training metrics
- [ ] Implement early stopping
- [ ] Implement gradient clipping (?)
- [ ] Implement gradient accumulation (?)
- [ ] Implement mixed precision training (?)
- [ ] Implement distributed training (?)
- [ ] Implement model parallelism (?)
- [ ] Implement data parallelism (?)
- [ ] Implement pipeline parallelism (?)

### Observability improvements:

- [x] Implement Tensorboard
- [ ] Add tracking of different test-training metrics (params, testing name, time). 
- [x] Implement training time metrics