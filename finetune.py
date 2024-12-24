import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm
from tiktoken import Encoding
import tiktoken
from data.RLHF.prepare import prepare_data
from model import GPTLMRewardModel, GPTLM

class PreferenceDataset(Dataset):
    def __init__(
        self,
        chosen_responses: List[str],
        rejected_responses: List[str],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        """
        Dataset for training a reward model on human preference data.

        Args:
            chosen_responses: List of preferred responses
            rejected_responses: List of non-preferred responses
            tokenizer: Tokenizer for processing text
            max_length: Maximum sequence length
        """
        self.chosen_responses = chosen_responses
        self.rejected_responses = rejected_responses
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.chosen_responses)

    def __getitem__(self, idx):
        # Tokenize both responses
        chosen = self.tokenizer(
            self.chosen_responses[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        rejected = self.tokenizer(
            self.rejected_responses[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen["input_ids"].squeeze(),
            "chosen_attention_mask": chosen["attention_mask"].squeeze(),
            "rejected_input_ids": rejected["input_ids"].squeeze(),
            "rejected_attention_mask": rejected["attention_mask"].squeeze(),
        }


class CustomRewardTrainer:
    def __init__(
        self,
        model: GPTLMRewardModel,
        tokenizer: AutoTokenizer,
        learning_rate: float = 1e-5,
        device: str = "tpu",
    ):
        """
        Initialize the reward model trainer.

        Args:
            model: Instance of GPTLMRewardModel
            tokenizer: Tokenizer for processing text
            learning_rate: Learning rate for training
            device: Device to use for training (default: tpu)
        """
        self.device = device
        import torch_xla.core.xla_model as xm
        self.device = xm.xla_device()
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        # Only optimize the reward head parameters
        self.optimizer = torch.optim.AdamW(
            self.model.reward_head.parameters(), 
            lr=learning_rate
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Perform one training step.
        """
        self.model.train()

        # Get chosen and rejected scores
        chosen_rewards = self.model(
            batch["chosen_input_ids"].to(self.device)
        )
        rejected_rewards = self.model(
            batch["rejected_input_ids"].to(self.device)
        )

        # Computing the loss: we want chosen_rewards > rejected_rewards
        loss = -torch.nn.functional.logsigmoid(
            chosen_rewards - rejected_rewards
        ).mean()

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(
        self,
        train_dataset: PreferenceDataset,
        batch_size: int = 8,
        num_epochs: int = 3,
        eval_dataset: PreferenceDataset = None,
    ) -> List[float]:
        """
        Train the reward model.

        Args:
            train_dataset: Dataset of preference pairs for training
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            eval_dataset: Optional evaluation dataset

        Returns:
            losses: List of training losses per epoch
        """
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        losses = []

        for epoch in range(num_epochs):
            epoch_losses = []

            for batch in tqdm(
                train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"
            ):
                loss = self.train_step(batch)
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}"
            )

            if eval_dataset:
                eval_accuracy = self.evaluate(eval_dataset)
                print(f"Evaluation Accuracy: {eval_accuracy:.4f}")

        return losses

    def evaluate(
        self, eval_dataset: PreferenceDataset, batch_size: int = 8
    ) -> float:
        """
        Evaluate the reward model on a validation set.
        """
        self.model.eval()
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                chosen_rewards = self.model(
                    batch["chosen_input_ids"].to(self.device)
                )
                rejected_rewards = self.model(
                    batch["rejected_input_ids"].to(self.device)
                )

                correct += (chosen_rewards > rejected_rewards).sum().item()
                total += len(chosen_rewards)

        return correct / total

    def save_model(self, path: str):
        """Save the reward model."""
        torch.save(self.model.state_dict(), path)


if __name__ == "__main__":
    # Disable XLA
    import os
    os.environ['XLA_USE_BF16'] = "0"
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = "0"
    
    # First load the pretrained weights with specific config to avoid XLA checks
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        "polrf/GPT2-GQA-RoPe",
        trust_remote_code=True,
        attn_implementation="eager"  # Explicitly use eager implementation instead of SDPA
    )
    
    # Initialize your updated GPTLM class
    new_model = GPTLM(pretrained_model.config)
    # Load the pretrained weights into your model
    new_model.load_state_dict(pretrained_model.state_dict(), strict=False)
    
    # Create the reward model using your updated GPTLM
    reward_model = GPTLMRewardModel(new_model)
    
    # Rest of the initialization
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    trainer = CustomRewardTrainer(reward_model, tokenizer)
    data = prepare_data()
    train_dataset = PreferenceDataset(data["train"]["chosen"], data["train"]["rejected"], tokenizer)
    eval_dataset = PreferenceDataset(data["test"]["chosen"], data["test"]["rejected"], tokenizer)
    trainer.train(train_dataset, eval_dataset=eval_dataset)




