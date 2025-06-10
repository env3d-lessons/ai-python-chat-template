from chat import complete
print(complete("<|im_start|>user\nWhat is the capital of Canada?<|im_end|>\n<|im_start|>assistant", temperature=1.0, max_new_tokens=200))

# from chat import get_next_token_dictionary

# def complete(prompt, max_new_tokens: int = 1):
#     response = prompt
#     for _ in range(max_new_tokens):
#         next_tokens = get_next_token_dictionary(response)
#         if not next_tokens:
#             break
#         # Sort tokens by probability and select the least likely one
#         next_token = sorted(next_tokens, key=lambda x: x['probability'])[0]['token']
#         response += next_token
#     return response

# print(complete("The capital of Canada is", max_new_tokens=1))