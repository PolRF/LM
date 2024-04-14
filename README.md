# Language Models Implementation Repository

## Project Overview

This repository is mainly created for educational purposes (mainly my own), with an emphasis on the practical implementation of state-of-the-art (SOTA) language models papers, utilizing the PyTorch library.

The main goal of this project is to provide a comprehensive and detailed implementation of the most recent and popular language models, such as GPT-2, Llama 2, Mistral, and others, as well as to provide a detailed explanation of the underlying concepts and mechanisms of these models.

## Future Work and TODO's

The following are among the planned future works and 'To Do' items for this project:

Model/Architecture improvements:

- [x] GPT-2
- [] Implement GeLU instead of RELU
- [] Take a look to Flash Attention (https://arxiv.org/pdf/2205.14135.pdf)
- [] Implement RoPE
- [] Implement Mixture of Experts
- [] Mamba
- [] Jamba
- [] Implement the Transformer-XL
- [] Implement Infinite attention (https://arxiv.org/pdf/2404.07143.pdf)
- [] Study if Infinite attention can be implemented on top of pre-trained models like Mixtral of Experts

Fine-tuning improvements:

- [] Load pre-trained models
- [] Implement LoRA
- [] Implement QLoRA

Training improvements:

- [] New SOTA AdamW optimizer
- [] Dynamic learning rate
- [] Implement checkpoints
- [] Better visualization of training metrics
- [] Implement early stopping
- [] Implement gradient clipping (?)
- [] Implement gradient accumulation (?)
- [] Implement mixed precision training (?)
- [] Implement distributed training (?)
- [] Implement model parallelism (?)
- [] Implement data parallelism (?)
- [] Implement pipeline parallelism (?)
