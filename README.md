# Language Models Implementation Repository

## Project Overview

This repository is mainly created for educational purposes (mainly my own), with an emphasis on the practical implementation of state-of-the-art (SOTA) language model papers, using the PyTorch library.

The main goal of this project is to provide a comprehensive and detailed implementation of the most recent and popular language models, such as GPT-2, Llama 2, Mistral, and others, as well as to provide a detailed explanation of the underlying concepts and mechanisms of these models.

## Achieved Goals
  Improved GPT-2 model:
  - Context: Implemented the GPT-2 model with some major improvements: GeLU activation function, RoPE (Relative Positional Encoding), GQA, flash attention and learning rate decay while training.
  - Model params: 123M
  - Results: 3.08 val loss after 2000 steps and a lowest training loss of 2.88. With further training, 3.029 validation loss was achieved. Minimum training loss at 2.71
  - Conclusions: A improvement from the original GPT-2 according to the Karpathy's nanogpt baseline of 3.11 train loss and 3.12 val loss. The loss was still decaying when the training achieved 5000 steps so I could keep training in order to achieve the 2.85 benchmark of a finetuned gpt-2 just to make sure that with less params, rotary positional embeddings and other changes could improve the base gpt-2 model.
  - Further improvements: Change hyperparams to improve the model and keep training it until we can outperform the finetuned baseline of 2.85 val loss. We can add GQA in order to upscale the model and achieve better results.

## Results:
The results of the experiments can be found in the [TESTS.md](TESTS.md) file.

## Future Work and TODO's

The following are among the planned future works and 'To Do' items for this project:

### Model/Architecture improvements:

- [x] GPT-2
- [x] Implement GeLU instead of RELU
- [x] Combine the `Head` and `MultiHeadAttention` into one class that processes all the heads in parallel, treating the heads as another batch dimension.
- [x] Take a look to Flash Attention (https://arxiv.org/pdf/2205.14135.pdf)
- [x] Implement RoPE
- [x] Implement weight sharing between token embedding and the last lm_head layer
- [x] Improve the RoPE implementation to apply the rotation to both the queries and keys at the same time
- [x] Implement weight tying (https://arxiv.org/pdf/1608.05859.pdf)
- [x] Scale the model to visualize better the improvements
- [x] Implement Grouped Query Attention (GQA)
- [ ] Implement "model surgery to decrease the block size"
- [ ] Implement KV-cache
- [ ] Implement Mixture of Experts (Mixtral)
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
- [ ] Try KAN layer instead of MLP (FFN) (paper:https://arxiv.org/abs/2404.19756, code: https://github.com/KindXiaoming/pykan)
- [ ] Implement model scaling
- [ ] CLLMs (multi token prediction)
- [ ] Read https://arxiv.org/abs/2405.17247
- [ ] Distillation
- [ ] Implement Alibi (https://arxiv.org/pdf/2405.17247.pdf)

### Fine-tuning improvements:

- [x] Load pre-trained models
- [ ] Implement LoRA
- [ ] Implement QLoRA

### Training improvements:

- [x] Implement flash attention to speed up training
- [x] Use a larger dataset to avoid overfitting
- [x] Dynamic learning rate
- [x] Implement model checkpoint saving for resuming training
- [x] Better visualization of training metrics
- [x] Config different precision for different parts of the model
- [x] Compile the model
- [x] Config optimizer (to be more efficient)
- [x] Add dtype as parameter for training
- [x] Implement gradient clipping
- [x] Implement gradient accumulation (micro-batching)
- [x] Implement mixed precision training
- [x] Take a look at chinchilla (https://arxiv.org/pdf/2205.14135.pdf)
- [ ] New SOTA AdamW optimizer
- [ ] Implement caching for the attention mechanism (across the model)
- [ ] Implement gradient checkpointing to reduce memory usage
- [ ] Implement early stopping
- [ ] Add pytorch profiler to debug bottlenecks
- [ ] Take a look at pytorch lightning
- [ ] Implement distributed training (?)
- [ ] Implement model parallelism (?)
- [ ] Implement data parallelism (?)
- [ ] Implement pipeline parallelism (?)

### Observability improvements:

- [x] Implement Tensorboard
- [x] Add tracking of different test-training metrics (params, testing name, time). 
- [x] Add gpu usage metrics
- [x] Augment the logging of the training metrics with wandb (instead of tensorboard)
- [x] Implement training time metrics

### Code improvements:
- [ ] Clean the training code