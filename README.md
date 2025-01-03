# Language Models Implementation Repository

## Project Overview

This repository is mainly created for educational purposes (mainly my own), with an emphasis on the practical implementation of state-of-the-art (SOTA) language model papers, using the PyTorch library. The whole project is inspired by the NanoGPT project of Andrej Karpathy (https://github.com/karpathy/nanoGPT).

The main goal of this project is to provide a comprehensive and detailed implementation of the most recent and popular language models, such as GPT-2, Llama 2, Mistral, and others, as well as to provide a detailed explanation of the underlying concepts and mechanisms of these models.

## Experiments:
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
- [x] Implement weight tying (https://arxiv.org/pdf/1608.05859.pdf)
- [x] Implement Grouped Query Attention (GQA)
- [x] Check training with autocast disabled for apply_rope
- [x] Implement Alibi (https://arxiv.org/pdf/2405.17247.pdf)
- [x] Implement KV-cache
- [x] Implement Mixture of Experts 500M params
- [x] Train a model with 2.3B (GPT-XL)
- [x] Integrate with hugging face AutoConfig and AutoModel
- [x] Implement multiple Rope theta values for different sequence lengths and see how it affects the model
- [x] Compare MoE with GPT (same size)
- [ ] Improve RoPE implementation using triton
- [ ] Implement Swiglu
- [ ] Change gpt2 tokenizer to Llama 3.1 tokenizer
- [ ] Extend context of a trained model to 128K through LongRope and finetuning if needed (https://arxiv.org/pdf/2402.13753)
- [ ] Mamba
- [ ] Jamba
- [ ] Implement SAMBA
- [ ] Implement the Transformer-XL
- [ ] Implement linear transformer
- [ ] Implement Infinite attention (https://arxiv.org/pdf/2404.07143.pdf)
- [ ] Study if Infinite attention can be implemented on top of pre-trained models like Mixtral of Experts
- [ ] BitNet: Scaling 1-bit Transformers for Large Language Models (https://arxiv.org/pdf/2310.11453.pdf)
- [ ] Implement model scaling
- [ ] CLLMs (multi token prediction)
- [ ] Read https://arxiv.org/abs/2405.17247
- [ ] Distillation
- [ ] Submit LLM to hugging face open llm leaderboard https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
- [ ] Implement entropyx sampler

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
- [x] Use Fineweb dataset instead of Openwebtext
- [x] Implement some optimizations to speed up training
- [x] Add pytorch profiler
- [x] Implement distributed data parallelism
- [x] Implement some evalutations --> Hellaswag
- [ ] eleuther harness reports
- [ ] Check param initialization
- [ ] Implement early stopping
- [ ] Take a look at pytorch lightning
- [ ] Implement model parallelism
- [ ] Implement pipeline parallelism

### Fine tuning:

- [ ] Reward model fine-tuning
- [ ] Supervised fine-tuning
- [ ] RL fine-tuning (PPO)
- [ ] Layer wise learning rate optimization
- [ ] Different model freezing strategies
- [ ] Implement LoRA
- [ ] Implement QLoRA

### Data:
- [x] Train with the Fineweb dataset (edu) 10B
- [ ] Train using an instruction dataset
- [ ] Read Efficient Training of Language Models to Fill in the Middle https://arxiv.org/pdf/2207.14255
- [ ] Read WizardLM (https://arxiv.org/pdf/2304.12244)

### Observability improvements:

- [x] Implement Tensorboard
- [x] Add tracking of different test-training metrics (params, testing name, time). 
- [x] Augment the logging of the training metrics with wandb (instead of tensorboard)
