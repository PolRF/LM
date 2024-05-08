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
    1. Minimal difference between train and validation loss which indicates that the larger dataset helps avoid overfitting.
    2. Larger dataset increase the number of steps to achive similar tran-val losses.
    3. Increasing the gpu helped with time per step from 1.23s recurrent to a decaying time per step from 0.75 to 0.68 (not sure why the time decays along the steps).
- Added microbatching:
  - Context: The model (gpt2-160M params) was too large for the gpu memory (24gb). I added microbatching to avoid this issue. Now batch size is 12 and gradient accumulation steps is 40.
  - Results: step 500: train loss 5.7003, val loss 5.7132, time (s): 23.80275, full time: 11990.43904. 
  - Conclusions: The model now is able to run in a single L4 gpu (24gb ram) only using 13gb. But the time per step is now 23s. This is because the model is now processing 40 steps before updating the weights. This is a tradeoff between memory and time. We have to lower the time per step and augment the usage of the gpu memory. Logging the info per step, I can see that the time per step decays from 138s to 67s in just 9 steps. This can be due to the fact that the model is learning and the gradients are getting smaller. Lets try to load the gpt-2 checkpoints and see if the model starts with a lower time per step.
- From pretrained gpt-2:
  - Context: In the previous iterations I saw that the time per step was too high but it was decaying along the steps. I wanted to test if the pretrained model starts with a lower time per step and the gradients update faster (as they are expected to get lower). Also remember that this model is 160M params with the same architecture as GPT-2 but using RoPE instead of the positional embeddings.
  - Results: step 500: train loss 3.6311, val loss 3.6341, time (s): 24.32832, full time: 12256.71348
  - Conclusions: The training started with a time per step similar to the raw model (108s in this case). Nevertheless, the time per step decayed much faster to 33s by step 9. By the step 30 the model was achieving a validation loss of 5.3. This is a good sign that the model is learning faster than the raw model. By step 500, the loss was 3.63, much lower than the training without the pretrained model. I observed that the time per step was around 24s and the loss was around 3.6, maybe implementing decaying learning rate we should achieve a better loss. Also, the model is using 13gb of the gpu memory, I should increase the gpu usage tu fill the 24gb of the L4 gpu.
- Dynamic learning rate pretrained gpt-2:
  - Context: Implemented a dynamic learning rate to decay the learning rate along the steps. The learning rate starts at 6e-4 and decays to a minimum of 6e-5.
  - Results: step 500: train loss 3.321, val loss 3.366, time (s): 24.8, full time: -
  - Conclusions: The implementation of the decaying lr has been successful since the model is learning faster and achieving a lower loss. The model trained until step 1281 (the gpu credits were over). The lowest training loss was 3.051. The model was using 15gb of the gpu memory. I should implement checkpoints to don't lose this training when I run out of gpu credits, more logging with wandb and change the hyperparams to use the full 24gb of the gpu.
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
- [ ] Research (and implement?) weight tying (https://arxiv.org/pdf/1608.05859.pdf)
- [ ] Implement "model surgery to decrease the block size"
- [x] Scale the model to visualize better the improvements
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
- [ ] Try KAN layer instead of MLP (FFN) (paper:https://arxiv.org/abs/2404.19756, code: https://github.com/KindXiaoming/pykan)
- [ ] Implement model scaling

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
- [x] Dynamic learning rate
- [ ] Implement gradient checkpointing to reduce memory usage
- [x] Implement model checkpoint saving for resuming training
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
- [ ] Augment the logging of the training metrics with wandb (instead of tensorboard)
- [x] Implement training time metrics

### Code improvements:
- [ ] Clean the training code