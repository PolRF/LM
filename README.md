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
  
## Future Work and TODO's

The following are among the planned future works and 'To Do' items for this project:

### Model/Architecture improvements:

- [x] GPT-2
- [x] Implement GeLU instead of RELU
- [ ] Combine the `Head` and `MultiHeadAttention` into one class that processes all the heads in parallel, treating the heads as another batch dimension.
- [ ] Take a look to Flash Attention (https://arxiv.org/pdf/2205.14135.pdf)
- [ ] Implement RoPE
- [ ] Implement Mixture of Experts
- [ ] Mamba
- [ ] Jamba
- [ ] Implement the Transformer-XL
- [ ] Implement Infinite attention (https://arxiv.org/pdf/2404.07143.pdf)
- [ ] Study if Infinite attention can be implemented on top of pre-trained models like Mixtral of Experts
- [ ] BitNet: Scaling 1-bit Transformers for Large Language Models (https://arxiv.org/pdf/2310.11453.pdf)

### Fine-tuning improvements:

- [ ] Load pre-trained models
- [ ] Implement LoRA
- [ ] Implement QLoRA

### Training improvements:

- [ ] New SOTA AdamW optimizer
- [ ] Implement flash attention to speed up training
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

- [ ] Implement Tensorboard
- [ ] Implement training time metrics