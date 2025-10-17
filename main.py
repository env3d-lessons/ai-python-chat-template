from chat import get_top_tokens, complete
import random

def complete_least_likely(prompt, max_tokens=1):
    return None

def complete_most_likely(prompt, max_tokens=1):
    return None

def complete_random(prompt, max_tokens=1):
    return None

def custom_template_chat(prompt):
    # Using multiple line string to create a chat template input
    template = f"""<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
    return complete(template)

if __name__ == "__main__":
    # Test call for complete_least_likely
    print("Testing complete_least_likely:")
    print(complete_least_likely("The capital of Canada is", max_tokens=1))

    # Test call for complete_most_likely
    print("\nTesting complete_most_likely:")
    print(complete_most_likely("The capital of Canada is", max_tokens=1))

    # Test call for complete_random
    print("\nTesting complete_random:")
    print(complete_random("The capital of Canada is", max_tokens=1))

    # Test call for custom_template_chat
    print("\nTesting custom_template_chat:")
    print(custom_template_chat("Hello"))