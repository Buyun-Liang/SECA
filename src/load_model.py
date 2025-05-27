import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
from config import OPENAI_API_KEY, LLAMA3_8B, QWEN2_5_7B, QWEN2_5_14B, LLAMA2_13B, LLAMA3_3B
import time
import random

def load_target_llm(model_type):

    # Model checkpoint path
    model_map = {
        'llama3_8b': LLAMA3_8B,
        'llama3_3b': LLAMA3_3B,
        'llama2_13b': LLAMA2_13B,
        'qwen2_5_7b': QWEN2_5_7B,
        'qwen2_5_14b': QWEN2_5_14B,
        'gpt_4_1_nano': "gpt-4.1-nano-2025-04-14",
        'gpt_4o_mini': 'gpt-4o-mini-2024-07-18',
    }

    try:
        model_name = model_map[model_type]
    except KeyError:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Loading target LLM: {model_type}")

    if 'gpt' in model_type:
        return model_name, None

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        low_cpu_mem_usage=True
    ) 

    model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto")

    return model, tokenizer


class GPT():
    '''
    Ref: https://github.com/patrickrchao/JailbreakingLLMs
    '''
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20

    def __init__(self, model_name):
        self.model_name = model_name
        print(f"Using OpenAI model: {self.model_name}")

    def generate(self, 
                full_prompt,
                max_new_tokens: int, 
                temperature: float,
                top_p: float = 1.0,
                frequency_penalty: float = 0.0,
                presence_penalty: float = 0.0):
        '''
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                client = openai.OpenAI(api_key = OPENAI_API_KEY)
                seed_value = random.randint(1, 1_000_000)  # Different seed for each request

                response = client.chat.completions.create(
                            model = self.model_name,
                            messages = full_prompt,
                            max_tokens = max_new_tokens,
                            temperature = temperature,
                            top_p = top_p,
                            timeout = self.API_TIMEOUT,
                            seed = seed_value,
                            frequency_penalty = frequency_penalty,
                            presence_penalty = presence_penalty,
                            )
                # output = response["choices"][0]["message"]["content"]
                output = response.choices[0].message.content
                break

            except openai.APIError as e:
                #Handle API error here, e.g. retry or log
                print(f"OpenAI API returned an API Error: {e}")
                time.sleep(self.API_RETRY_SLEEP)
            except openai.APIConnectionError as e:
                #Handle connection error here
                print(f"Failed to connect to OpenAI API: {e}")
                time.sleep(self.API_RETRY_SLEEP)
            except openai.RateLimitError as e:
                #Handle rate limit error (we recommend using exponential backoff)
                print(f"OpenAI API request exceeded rate limit: {e}")
                time.sleep(self.API_RETRY_SLEEP)
        
            time.sleep(self.API_QUERY_SLEEP)
        return output 