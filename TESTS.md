### LogBook

- GPT:
  - Context: First training on Lightning.ai with T4 GPU. Small models, no expectation of good results, just to test the code.
  - Dataset: The Spanish Bible (bible_es.txt)
  - Params: 2 layers, 2 heads, 384 embedding size, 50304 vocab size (gpt2 tokenizer), 6e-5 learning rate, 256 block size, 64 batch size.

- Openweb dataset:
  - Context: New dataset with a much larger number of tokens to minimize the disparity between training and val loss. Also, I started using a Lightning L4 GPU with 24GB of RAM. With no changes to the hyperparameters, the training memory usage is around 18.5GB.
  - Dataset: Openwebtext
  - Params: 2 layers, 2 heads, 384 embedding size, 50304 vocab size (gpt2 tokenizer), 6e-5 learning rate, 256 block size, 64 batch size. Total: 42.31M params.
  - Conclusions: The results are what I expected. 
    1. Minimal difference between train and validation loss, which indicates that the larger dataset helps avoid overfitting.
    2. Larger dataset increases the number of steps to achieve similar train-val losses.
    3. Increasing the GPU helped with time per step, from 1.23s recurrent to a decaying time per step from 0.75 to 0.68 (not sure why the time decays along the steps).

- Added microbatching:
  - Context: The model (gpt2-123M params) was too large for the GPU memory (24GB). I added microbatching to avoid this issue. Now batch size is 12 and gradient accumulation steps is 40.
  - Conclusions: The model is now able to run on a single L4 GPU (24GB RAM) using only 13GB. But the time per step is now 23s. This is because the model is now processing 40 steps before updating the weights. This is a tradeoff between memory and time. We have to lower the time per step and increase the usage of the GPU memory. Logging the info per step, I can see that the time per step decays from 138s to 67s in just 9 steps. This could be due to the fact that the model is learning and the gradients are getting smaller. Let's try to load the gpt-2 checkpoints and see if the model starts with a lower time per step.

- From pretrained gpt-2 (but with RoPE):
  - Context: In the previous iterations, I saw that the time per step was too high but it was decaying along the steps. I wanted to test if the pretrained model starts with a lower time per step and if the gradients update faster (as they are expected to get lower). Also remember that this model has 123M params with the same architecture as GPT-2 but using RoPE instead of the positional embeddings.
  - Results: step 500: train loss 3.6311, val loss 3.6341, time (s): 24.32832, full time: 12256.71348
  - Conclusions: The training started with a time per step similar to the raw model (108s in this case). Nevertheless, the time per step decayed much faster to 33s by step 9. By step 30, the model was achieving a validation loss of 5.3. This is a good sign that the model is learning faster than the raw model. By step 500, the loss was 3.63, much lower than the training without the pretrained model. I observed that the time per step was around 24s and the loss was around 3.6; maybe implementing decaying learning rate we should achieve a better loss. Also, the model is using 13GB of the GPU memory; I should increase the GPU usage to fill the 24GB of the L4 GPU.

- Dynamic learning rate pretrained gpt-2:
  - Context: Implemented a dynamic learning rate to decay the learning rate along the steps. The learning rate starts at 6e-4 and decays to a minimum of 6e-5.
  - Results: step 500: train loss 3.321, val loss 3.366, time (s): 24.8, full time: -
  - Conclusions: The implementation of the decaying learning rate has been successful since the model is learning faster and achieving a lower loss. The model trained until step 1281 (when the GPU credits ran out). The lowest training loss was 3.051. The model was using 15GB of the GPU memory. I should implement checkpoints to avoid losing this training when I run out of GPU credits, add more logging with wandb, and change the hyperparameters to use the full 24GB of the GPU.

- Improved training script:
  - Context: I implemented different things to improve the training. Now we compile the model. We use mixed precision with the proper gradient scaler. Also, I implemented the wandb logger to keep track of the training.
  - Results: The time per step has decreased to 8-10 seconds thanks to compilation and mixed precision. Now the model can compare to Karpathy's initial setup. I achieved 3.387 loss training the GPT-2 model from scratch with RoPE and dynamic learning rate. The model was using 26GB of GPU (A6000). 
  - Conclusions: The training was as expected; if we could train for more than 60k iterations, we could achieve a lower loss (around 2.85). The time per step was 1.8s.
  - Final setup: 60k iterations, 6e-4 to 6e-5 decaying learning rate, 2k warmup steps, 0.1 weight decay, 12 batch size, and 5 gradient accumulation steps (to match the 0.5M tokens of the original setup, we should accumulate 40 steps). 1.0 gradient clip. Also, I used the 50304 vocab size instead of 50257 and 0.0 dropout.
  - Further considerations: Before the modifications of the training script, I was able to achieve a better loss of 3.08 by pretraining gpt-2 after 2 days of training. With the current script, I achieved 3.38 with 31h of training but with more stable training and completely from scratch, not using pretrained weights.
  - Additional comments: I could continue training the model (maybe I will) until I achieve the 2.85 benchmark of a finetuned gpt-2 just to make sure that with fewer params, rotary positional embeddings, and other changes, we could improve the base gpt-2 model. 

- GPT-2 RoPE + GQA:
  - Context: Implemented GQA (Grouped Query Attention) to improve the model. The model was trained with 16 query heads and 4 key-value heads. 
  - Model params: 113M
  - Results: The model trained much faster than the best result achieved before. The model achieved better validation and training loss not only faster but also with fewer steps. The time per step was reduced from 1.8s to 0.8s, and the model was using 14GB vs 26GB of the GPU memory. With almost 24h of training, the model achieved a validation loss of 3.305 vs 3.507. 
  - Conclusions: Significant improvement not only in the training time but also in the achieved loss with fewer trainable params (113M vs 123M).

- GPT-2 RoPE + GQA + 8xH100:
  - I could train the model on a cluster of 8xH100 GPUs. The model achieved a better Hellaswag evaluation result.
  - Context: Implemented the GPT-2 model with some major improvements: GeLU activation function, RoPE (Rotary Positional Encoding), GQA, flash attention, and learning rate decay while training. 14h of training (250k steps).
  - Dataset: Fineweb dataset (edu) 10B
  - Infra: AWS p5.48xlarge instance (8xH100 GPUs)
  - Model params: 113M
  - Results: Hellaswag evaluation accuracy of 32.03%. Val loss: 2.978, train loss: 2.787
  - Conclusions: Improved Andrej Karpathy's benchmark of 29.55% to 32.03% on Hellaswag eval (repo) with fewer params.

- GQA + RoPE + MoE:
  - Context: Implemented Mixture of Experts to improve the model. The model was trained with 16 query heads and 4 key-value heads, 8 experts, and 2 experts per token. 500M params in total.
  - Dataset: Fineweb dataset (edu) 10B
  - Training: 220k steps. +35h.
  - Infra: AWS p5.48xlarge instance (8xH100 GPUs)
  - Model params: 500M
  - Results: Hellaswag evaluation accuracy of 39.54%. Val loss: 2.696, train loss: 2.485

- GQA + RoPE vs MoE:
  - Context: Trained a GPT model without experts but same number of params than the MoE model.
  - Dataset: Fineweb dataset (edu) 10B
  - Training: 220k steps. 35h.
  - Infra: AWS p5.48xlarge instance (8xH100 GPUs)
  - Model params: 500M
  - Results: Val loss: 2.64961.
  - Conclusions: The model with GQA + RoPE achieved a better results than the MoE model (per time and steps). 

- 2.3B model:
  - Dataset: Fineweb dataset (edu) 10B
  - Training: 85k steps. 60h
  - Moldel params: 2.3B.
  - Results: Val loss: 2.5247.
  - Conclusions: Trained a 2.3B model. Nothing to add. Scale is all you need. Validation loss was better per time and steps.
  
- Multiple trainings to study the impact of theta RoPE param on different sequence lengths:
  Trained 113M model with 1024, 4096, and 8192 sequence lengths, each with 10k, 100k, 500k, 2M, and 10M theta param. 50-200k steps each. All training weights are available on Hugging Face.

  TODO: 
  - Publish a full white paper (or Twitter thread) with the results.
