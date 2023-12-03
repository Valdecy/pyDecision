###############################################################################

# Required Libraries
import openai  
import os

###############################################################################

# Function: Ask GPT Rank
def ask_chatgpt_corr(ranks, char_limit = 4097, api_key = 'your_api_key_here', query = 'which methods are more similar?', model = 'text-davinci-003', max_tokens = 2000, n = 1, temperature = 0.8):
    flag                     = 0
    os.environ['OPENAI_KEY'] = api_key
    corpus                   = ranks.to_string(index = False)
    context                  = ' knowing that for the given table, it shows the correlation values between MCDA methods'
    prompt                   = query + context + ':\n\n' + f'{corpus}\n'
    prompt                   = prompt[:char_limit]
    
    ##############################################################################
   
    def version_check(major, minor, patch):
        try:
            version                   = openai.__version__
            major_v, minor_v, patch_v = [int(v) for v in version.split('.')]
            if ( (major_v, minor_v, patch_v) >= (major, minor, patch) ):
                return True
            else:
                return False
        except AttributeError:
            return False
    
    if (version_check(1, 0, 0)):
        flag = 1
    else:
        flag = 0
    
    ##############################################################################
        
    def query_chatgpt(prompt, model = model, max_tokens = max_tokens, n = n, temperature = temperature):
        if (flag == 0):
          try:
              response = openai.ChatCompletion.create(model = model, messages = [{'role': 'user', 'content': prompt}], max_tokens = max_tokens)
              response = response['choices'][0]['message']['content']
          except:
              response = openai.Completion.create(engine = model, prompt = prompt, max_tokens = max_tokens, n = n, stop = None, temperature = temperature)
              response = response.choices[0].text.strip()
        else:
          try:
            client   = openai.OpenAI(api_key = api_key)
            response = client.chat.completions.create(model = model, messages = [{'role': 'user', 'content': prompt}], max_tokens = max_tokens)
            response = response.choices[0].message.content
          except:
            client   = openai.OpenAI(api_key = api_key)
            response = client.completions.create( model = model, prompt = prompt, max_tokens = max_tokens, n = n, stop = None, temperature = temperature)
            response = response.choices[0].text.strip()
        return response
    
    ##############################################################################
    
    analyze        = query_chatgpt(prompt)
    print('Number of Characters: ' + str(len(prompt)))
    return analyze

# Function: Ask GPT Rank
def ask_chatgpt_rank(ranks, char_limit = 4097, api_key = 'your_api_key_here', query = 'which methods are more similar?', model = 'text-davinci-003', max_tokens = 2000, n = 1, temperature = 0.8):
    flag                     = 0
    os.environ['OPENAI_KEY'] = api_key
    corpus                   = ranks.to_string(index = False)
    context                  = ' knowing that for the given outranking table, the columns represent MCDA methods and the rows alternatives. Each cell indicates the rank of the alternatives (1 = 1st position, 2 = 2nd position, and so on) '
    prompt                   = query + context + ':\n\n' + f'{corpus}\n'
    prompt                   = prompt[:char_limit]
    
    ##############################################################################
   
    def version_check(major, minor, patch):
        try:
            version                   = openai.__version__
            major_v, minor_v, patch_v = [int(v) for v in version.split('.')]
            if ( (major_v, minor_v, patch_v) >= (major, minor, patch) ):
                return True
            else:
                return False
        except AttributeError:
            return False
    
    if (version_check(1, 0, 0)):
        flag = 1
    else:
        flag = 0
    
    ##############################################################################
        
    def query_chatgpt(prompt, model = model, max_tokens = max_tokens, n = n, temperature = temperature):
        if (flag == 0):
          try:
              response = openai.ChatCompletion.create(model = model, messages = [{'role': 'user', 'content': prompt}], max_tokens = max_tokens)
              response = response['choices'][0]['message']['content']
          except:
              response = openai.Completion.create(engine = model, prompt = prompt, max_tokens = max_tokens, n = n, stop = None, temperature = temperature)
              response = response.choices[0].text.strip()
        else:
          try:
            client   = openai.OpenAI(api_key = api_key)
            response = client.chat.completions.create(model = model, messages = [{'role': 'user', 'content': prompt}], max_tokens = max_tokens)
            response = response.choices[0].message.content
          except:
            client   = openai.OpenAI(api_key = api_key)
            response = client.completions.create( model = model, prompt = prompt, max_tokens = max_tokens, n = n, stop = None, temperature = temperature)
            response = response.choices[0].text.strip()
        return response
    
    ##############################################################################    
    
    analyze        = query_chatgpt(prompt)
    print('Number of Characters: ' + str(len(prompt)))
    return analyze

# Function: Ask GPT Weights
def ask_chatgpt_weights(weights, char_limit = 4097, api_key = 'your_api_key_here', query = 'which methods are more similar?', model = 'text-davinci-003', max_tokens = 2000, n = 1, temperature = 0.8):
    flag                     = 0
    os.environ['OPENAI_KEY'] = api_key
    corpus                   = weights.to_string(index = False)
    context                  = ' knowing that for the given table, the columns represent MCDA methods and the rows criterion. Each cell indicates the weight of each criterion (the higher the value, the most important is the criterion) calculate by each MCDA method'
    prompt                   = query + context + ':\n\n' + f'{corpus}\n'
    prompt                   = prompt[:char_limit]
    
    ##############################################################################
   
    def version_check(major, minor, patch):
        try:
            version                   = openai.__version__
            major_v, minor_v, patch_v = [int(v) for v in version.split('.')]
            if ( (major_v, minor_v, patch_v) >= (major, minor, patch) ):
                return True
            else:
                return False
        except AttributeError:
            return False
    
    if (version_check(1, 0, 0)):
        flag = 1
    else:
        flag = 0
    
    ##############################################################################
        
    def query_chatgpt(prompt, model = model, max_tokens = max_tokens, n = n, temperature = temperature):
        if (flag == 0):
          try:
              response = openai.ChatCompletion.create(model = model, messages = [{'role': 'user', 'content': prompt}], max_tokens = max_tokens)
              response = response['choices'][0]['message']['content']
          except:
              response = openai.Completion.create(engine = model, prompt = prompt, max_tokens = max_tokens, n = n, stop = None, temperature = temperature)
              response = response.choices[0].text.strip()
        else:
          try:
            client   = openai.OpenAI(api_key = api_key)
            response = client.chat.completions.create(model = model, messages = [{'role': 'user', 'content': prompt}], max_tokens = max_tokens)
            response = response.choices[0].message.content
          except:
            client   = openai.OpenAI(api_key = api_key)
            response = client.completions.create( model = model, prompt = prompt, max_tokens = max_tokens, n = n, stop = None, temperature = temperature)
            response = response.choices[0].text.strip()
        return response
    
    ##############################################################################      
    
    analyze        = query_chatgpt(prompt)
    print('Number of Characters: ' + str(len(prompt)))
    return analyze

###############################################################################
