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
  - Results: Steps: 4600, training loss: 2.1387, validation loss: 3.1603
  - Conclusions: The model with RoPE is slightly better than the model with positional embeddings. The model with RoPE has a lower training loss and validation loss.
- Added flash attention:
  - Context: No major improvement (around 1.23 seconds per step). This is because flash attention is more useful for long sequences (not the case here).
  - Todo: We can test the difference when increasing the model size (and context length).
- Openweb dataset:
  - Context: New dataset with much larger amount of tokens to minimize the disparity between training and val loss. Also, I started using lightning L4 GPU with 24gb of ram. With no changes on the hyperparams, the training usage is around 18.5gb.
  - Dataset: Openwebtext
  - Params: 2 layers, 2 heads, 384 embedding size, 50304 vocab size (gpt2 tokenizer), 6e-5 learning rate, 256 block size, 64 batch size. Total: 42.31M params.
  - Results: step 4600: train loss 5.3644, val loss 5.3708, time (s): 0.68788, full time: 3220.8362
  - Conclusions: The results are what I expected. 
    1. Minimal difference between train and validation loss which indicates that the larger dataset helps avoiding overgitting. 
    2. Larger dataset increase the number of steps to achive similar tran-val losses.
    3. Increasing the gpu helped with time per step from 1.23s recurrent to a decaying time per step from 0.75 to 0.68 (not sure why the time decays along the steps).

## Future Work and TODO's

The following are among the planned future works and 'To Do' items for this project:

### Model/Architecture improvements:

- [x] GPT-2
- [x] Implement GeLU instead of RELU
- [x] Combine the `Head` and `MultiHeadAttention` into one class that processes all the heads in parallel, treating the heads as another batch dimension.
- [x] Take a look to Flash Attention (https://arxiv.org/pdf/2205.14135.pdf)
- [x] Implement RoPE
- [ ] Research (and implement?) weight tying (https://arxiv.org/pdf/1608.05859.pdf)
- [ ] Implement "model surgery to decrease the block size"
- [ ] Scale the model to visualize better the improvements
- [ ] Implement KV-cache
- [ ] Implement Mixture of Experts (Mixtral)
- [ ] Implement Grouped Query Attention (GQA)
- [ ] Sliding Window Attention (SWA)
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

- [x] Load pre-trained models
- [ ] Implement LoRA
- [ ] Implement QLoRA

### Training improvements:

- [ ] New SOTA AdamW optimizer
- [ ] Take a look at chinchilla (https://arxiv.org/pdf/2205.14135.pdf)
- [x] Implement flash attention to speed up training
- [x] Use a larger dataset to avoid overfitting
- [ ] Implement caching for the attention mechanism (across the model)
- [ ] Dynamic learning rate
- [ ] Implement gradient checkpointing to reduce memory usage
- [ ] Implement model checkpoint saving for resuming training
- [ ] Better visualization of training metrics
- [ ] Implement early stopping
- [ ] Implement gradient clipping (?)
- [x] Implement gradient accumulation (micro-batching)
- [ ] Implement mixed precision training (?)
- [ ] Implement distributed training (?)
- [ ] Implement model parallelism (?)
- [ ] Implement data parallelism (?)
- [ ] Implement pipeline parallelism (?)

### Observability improvements:

- [x] Implement Tensorboard
- [ ] Add tracking of different test-training metrics (params, testing name, time). 
- [ ] Add gpu usage metrics
- [x] Implement training time metrics