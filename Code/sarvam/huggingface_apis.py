# Model https://huggingface.co/sarvamai/sarvam-2b-v0.5
# Ref Notebook: https://colab.research.google.com/drive/1IZ-KJgzRAMr4Rm_-OWvWwnfTQwRxOknp?usp=sharing

from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
# import torch

model = AutoModelForCausalLM.from_pretrained("sarvamai/sarvam-2b-v0.5")
tokenizer = AutoTokenizer.from_pretrained("sarvamai/sarvam-2b-v0.5")
tokenizer.pad_token_id = tokenizer.eos_token_id
pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer, device="cuda", torch_dtype="bfloat16",
                              return_full_text=False)
gen_kwargs = {
    "temperature": 0.01,  # more definitive
    "repetition_penalty": 1.2,  # discourage the model from repeating recently generated tokens
    "max_new_tokens": 256,  # maximum number of tokens to generate, long answer
    "stop_strings": ["</s>", "\n\n"],  # if these tokens are seen, stop the generation
    "tokenizer": tokenizer,  # tokenizer of the model
    # "torch_dtype": torch.float16,
    # "low_cpu_mem_usage":True,
    # "device_map":"auto",
    # "load_in_8bit":True,
}  # seems to work best with these defaults


def gen(prompt):
    prompt = prompt.rstrip() # having a trailing space hurts the model generation; let us strip them
    output = pipe(prompt, **gen_kwargs)
    print(output[0]["generated_text"])


if __name__ == "__main__":
    gen("छत्रपती शिवाजी महाराजांच्या साम्राज्याची राजधानी :")
