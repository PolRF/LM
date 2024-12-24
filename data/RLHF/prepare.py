from datasets import load_dataset

def prepare_data():
    dataset = load_dataset("Anthropic/hh-rlhf")
    return dataset