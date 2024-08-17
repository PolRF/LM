from huggingface import GPTConfig, GPTModel

# from transformers import pipeline, AutoTokenizer


# tokenizer = AutoTokenizer.from_pretrained("gpt2")

GPTConfig.register_for_auto_class()
GPTModel.register_for_auto_class("AutoModel")
GPTModel.register_for_auto_class("AutoModelForCausalLM")
config = GPTConfig.from_pretrained("polrf/GPT2-GQA-RoPe")
model = GPTModel(config).from_pretrained("polrf/GPT2-GQA-RoPe")
# model.push_to_hub("polrf/GPT2-GQA-RoPe", safe_serialization=False)
# generator = pipeline(
#     "text-generation", model=model, tokenizer=tokenizer, trust_remote_code=True
# )
# print(generator("The future of AI is"))
