from huggingface import GPTConfig, GPTModel


GPTConfig.register_for_auto_class()
GPTModel.register_for_auto_class("AutoModel")
GPTModel.register_for_auto_class("AutoModelForCausalLM")
config = GPTConfig.from_pretrained("polrf/GPT2-GQA-RoPe")
model = GPTModel(config).from_pretrained("polrf/GPT2-GQA-RoPe")
model.push_to_hub("polrf/GPT2-GQA-RoPe", safe_serialization=False)
