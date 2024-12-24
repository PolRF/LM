import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the reward model trainer.

        Args:
            model: Instance of GPTLMRewardModel
            tokenizer: Tokenizer for processing text
            learning_rate: Learning rate for training
            device: Device to use for training
        """
        self.device = device
        self.model = model.to(device)
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


def train(rank, flags):
    # Load the pretrained weights
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        "polrf/GPT2-GQA-RoPe", trust_remote_code=True
    )
    config = pretrained_model.config
    device = xm.xla_device()
    
    # Initialize your updated GPTLM class
    new_model = GPTLM(config)
    new_model.load_state_dict(pretrained_model.state_dict(), strict=False)
    
    # Create the reward model using your updated GPTLM
    reward_model = GPTLMRewardModel(new_model)
    reward_model.to(device)
    
    # Rest of the initialization
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    trainer = CustomRewardTrainer(reward_model, tokenizer, device=device)
    
    # Prepare data
    data = prepare_data()
    train_dataset = PreferenceDataset(data["train"]["chosen"], data["train"]["rejected"], tokenizer)
    eval_dataset = PreferenceDataset(data["test"]["chosen"], data["test"]["rejected"], tokenizer)
    
    # Use XLA's parallel loader for data
    train_loader = pl.ParallelLoader(DataLoader(train_dataset, batch_size=flags['batch_size'], shuffle=True), [device]).per_device_loader(device)
    eval_loader = pl.ParallelLoader(DataLoader(eval_dataset, batch_size=flags['batch_size']), [device]).per_device_loader(device)
    
    # Training loop
    for epoch in range(flags['num_epochs']):
        epoch_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{flags['num_epochs']}"):
            loss = trainer.train_step(batch)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch + 1}/{flags['num_epochs']}, Average Loss: {avg_loss:.4f}")
        
        # Evaluation
        eval_accuracy = trainer.evaluate(eval_loader)
        print(f"Evaluation Accuracy: {eval_accuracy:.4f}")


if __name__ == "__main__":
    flags = {
        'batch_size': 8,
        'num_epochs': 3
    }
    xmp.spawn(train, args=(flags,), nprocs=4, start_method='fork')
    



