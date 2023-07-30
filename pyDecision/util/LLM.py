###############################################################################

# Required Libraries
import openai  

###############################################################################

# Function: Ask GPT Rank
def ask_chatgpt_rank(ranks, char_limit = 4097, api_key = 'your_api_key_here', query = 'which methods are more similar?', model = 'text-davinci-003', max_tokens = 2000, n = 1, temperature = 0.8):
    def query_chatgpt(prompt, model = model, max_tokens = max_tokens, n = n, temperature = temperature):
        try: 
            response = openai.ChatCompletion.create(
                                                    model      = model,
                                                    messages   = [{'role': 'user', 'content': prompt}],
                                                    max_tokens = max_tokens
                                                    )
            response = response['choices'][0]['message']['content']
        except:
            response = openai.Completion.create(
                                                engine      = model,
                                                prompt      = prompt,
                                                max_tokens  = max_tokens,
                                                n           = n,
                                                stop        = None,
                                                temperature = temperature
                                                )
            response = response.choices[0].text.strip()
        return response
    corpus         = ranks.to_string(index = False)
    openai.api_key = api_key
    context        = ' knowing that for the given outranking table, the columns represent MCDA methods and the rows alternatives. Each cell indicates the rank of the alternatives (1 = 1st position, 2 = 2nd position, and so on) '
    prompt         = query + context + ':\n\n' + f'{corpus}\n'
    prompt         = prompt[:char_limit]
    analyze        = query_chatgpt(prompt)
    print('Number of Characters: ' + str(len(prompt)))
    return analyze

# Function: Ask GPT Weights
def ask_chatgpt_weights(weights, char_limit = 4097, api_key = 'your_api_key_here', query = 'which methods are more similar?', model = 'text-davinci-003', max_tokens = 2000, n = 1, temperature = 0.8):
    def query_chatgpt(prompt, model = model, max_tokens = max_tokens, n = n, temperature = temperature):
        try: 
            response = openai.ChatCompletion.create(
                                                    model      = model,
                                                    messages   = [{'role': 'user', 'content': prompt}],
                                                    max_tokens = max_tokens
                                                    )
            response = response['choices'][0]['message']['content']
        except:
            response = openai.Completion.create(
                                                engine      = model,
                                                prompt      = prompt,
                                                max_tokens  = max_tokens,
                                                n           = n,
                                                stop        = None,
                                                temperature = temperature
                                                )
            response = response.choices[0].text.strip()
        return response
    corpus         = weights.to_string(index = False)
    openai.api_key = api_key
    context        = ' knowing that for the given table, the columns represent MCDA methods and the rows criterion. Each cell indicates the weight of each criterion (the higher the value, the most important is the criterion) calculate by each MCDA method'
    prompt         = query + context + ':\n\n' + f'{corpus}\n'
    prompt         = prompt[:char_limit]
    analyze        = query_chatgpt(prompt)
    print('Number of Characters: ' + str(len(prompt)))
    return analyze

###############################################################################
