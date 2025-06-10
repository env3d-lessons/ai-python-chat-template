# Chat Completion basics

The original intention of LLM is to generated text for use in "autocomplete" 
applications.  Despite all the AI advances and behind all the fancy APIs and libraries,
the core of the LLM is still a text generator. 

I have provided a function called get_next_token which returns a list of
potential completion with associated probabilities, like this:

```python

# Put this code in main.py
from chat import get_next_token_dictionary

print(get_next_token_dictionary("The capital of Canada is "))
```

Now we run it:

```
$ python main.py
[{'token': ' Ottawa', 'probability': 0.7608394622802734}, 
{'token': '渥', 'probability': 0.0402972549200058}, 
{'token': ' located', 'probability': 0.026063064113259315}, 
{'token': ' Montreal', 'probability': 0.014869445003569126}, 
{'token': ' not', 'probability': 0.01464066095650196}, 
{'token': ' Toronto', 'probability': 0.011498438194394112}, 
{'token': ' the', 'probability': 0.006867176853120327}, 
{'token': ' in', 'probability': 0.006486607249826193}, 
{'token': ' __', 'probability': 0.005786129739135504}, 
{'token': ' Canada', 'probability': 0.005360356532037258}]
```

Technically, the next completion is called a "token".  A token can be a word, a part of 
a word, or even punctuation.

As you can see, the most likely next token is " Ottawa" with a probability of 0.76.  But an
LLM does not always return the most likely next token.  As programmers, we can control
this by setting a parameter called "temperature".  A temperature of 0.0 means the most likely
token is always returned.  A temperature of 1.0 means the next token is randomly selected
from the list of potential tokens, weighted by their probabilities.  The higher the temperature, the more random the next token is.  

The `complete` function in the `chat` module allows you to generate a full response
by passing in a prompt and a temperature. 

Here's how it works:

```python
from chat import complete
print(complete("The capital of Canada is", temperature=0.01, max_new_tokens=1))
```

When you run this code, it will generate a response based on the prompt "The capital of Canada is" with a very low temperature, meaning it will likely return the most probable next token.

```
$ python main.py 
The capital of Canada is Ottawa
```

## Exercise

Try changing the value of the `temperature` parameter to see how it affects the output.

You can also adjust the `max_new_tokens` parameter to control how many tokens you want to generate in the response.

Write you own complete function using the `get_next_token_dictionary` function.  
For example, following function always return the least likely token from the list of potential tokens returned by `get_next_token_dictionary`:

```python
from chat import get_next_token_dictionary
def complete(prompt, max_new_tokens: int = 1):
    response = prompt
    for _ in range(max_new_tokens):
        next_tokens = get_next_token_dictionary(response)
        if not next_tokens:
            break
        # Sort tokens by probability and select the least likely one
        next_token = sorted(next_tokens, key=lambda x: x['probability'])[0]['token']
        response += next_token
    return response

print(complete("The capital of Canada is", max_new_tokens=1))
```

# Chat Template

Since LLMs simply generated the next token, they are actually not very good 
at carrying on a conversation.  To make them more useful, scientists have developed
a technique called "instruction tuning" - it basically a fancy name for structuring the
training dataset to conform to a specific format.  The most common format is the "chat template".  Below are some example entries in such a dataset:

```
<|im_start|>user
What is the capital of Canada?<|im_end|>
<|im_start|>assistant
The capital of Canada is Ottawa.<|im_end|>
```

```
<|im_start|>user
What is the capital of Canada?<|im_end|>
<|im_start|>assistant
The capital of Canada is Ottawa.<|im_end|>
```

Noticed some special strings like `<|im_start|>user` and `<|im_end|>`.  
These are called special tokens and they help the LLM understand the structure of the
conversation.  So imagine the LLM is trained on a dataset with these special tokens, it
learns to recognize the pattern so if we give it the input prompt 
"<|im_start|>user\nWhat is the capital of Canada?<|im_end|>\n<|im_start|>assistant", it
will try to complete the response with what the dataset thinks is the most appropriate answer
from the assistant.

## Exercise

You can prove that it works by using the `complete` function with a chat template.

```python
from chat import complete

# Using multiple line string to create a chat template input
prompt = """
<|im_start|>user
What is the capital of Canada?<|im_end|>
<|im_start|>assistant
"""

print(complete(prompt, temperature=1.0, max_new_tokens=200))
```



When you run this code, it will generate a response based on the chat template input.

```