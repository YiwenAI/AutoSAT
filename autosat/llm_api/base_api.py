import openai


class BaseCallAPI():
    def __init__(self, api_base, api_key, model_name):
        self.api_base = api_base
        self.api_key = api_key
        openai.api_base = self.api_base
        openai.api_key = self.api_key
        self.model_name = model_name

    def load_prompt(self, file_dir):
        with open(file_dir, 'r') as file:
            prompt = file.read()
        return prompt

    def call_api(self, prompt, temperature):
        pass

class GPTCallAPI(BaseCallAPI):
    def __init__(self, api_base, api_key, model_name, stream):
        super(GPTCallAPI, self).__init__(api_base, api_key, model_name)
        self.stream = stream

    def call_api(self, prompt_file, temperature=0.2):
        with open(prompt_file, 'r') as file:
            prompt = file.read()
        response = openai.ChatCompletion.create(
            model=self.model_name,
            # model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a chatbot", },
                {"role": "user", "content": prompt}],
            temperature=temperature,
            stream=self.stream
        )

        if self.stream:
            result = ""
            try:
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        result += chunk.choices[0].delta.content
                        # print(chunk.choices[0].delta.content, end="")
            except:
                pass
        else:
            result = response["choices"][0]["message"]["content"]

        return result

    def call_api_prompt(self, prompt, temperature=0.2):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a chatbot", },
                {"role": "user", "content": prompt}],
            temperature=temperature,
            stream=self.stream
        )

        if self.stream:
            result = ""
            try:
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        result += chunk.choices[0].delta.content
                        # print(chunk.choices[0].delta.content, end="")
            except:
                pass
        else:
            result = response["choices"][0]["message"]["content"]

        return result

class LocalCallAPI(BaseCallAPI):
    def __init__(self, api_base, api_key, model_name):
        super(LocalCallAPI, self).__init__(api_base, api_key, model_name)

    def call_api(self, prompt_file,
                 temperature=0.2):
        stop_tokens = ["<|im_end|>"]
        system_prompt = "You are a chatbot"

        with open(prompt_file, 'r') as file:
            prompt = file.read()

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}],
                stop=stop_tokens)
        return response["choices"][0]["message"]["content"]

    def call_api_prompt(self, prompt,
                 temperature=0.2):
        stop_tokens = ["<|im_end|>"]
        system_prompt = "You are a chatbot"

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}],
                stop=stop_tokens,
                temperature=temperature,)

        return response["choices"][0]["message"]["content"]

# *----------- Fast llm call --------------------*
def fastllm(prompt , args):
    if args.llm_model[0:3] == "gpt":
        llm_api = GPTCallAPI(api_base=args.api_base,
                             api_key=args.api_key,
                             model_name=args.llm_model,
                             stream=False)
    elif args.llm_model == 'Qwen':
        llm_api = LocalCallAPI(api_base="http://172.26.1.16:31251/v1",
                               api_key="sk-",
                               model_name="modelscope/qwen/Qwen-72B-Chat")
    elif args.llm_model == 'llama':
        llm_api = LocalCallAPI(api_base="http://172.26.1.16:31251/v1",
                               api_key="sk-",
                               model_name="modelscope/modelscope/Llama-2-70b-chat-ms")
    elif args.llm_model == 'deepseek':
        llm_api = LocalCallAPI(api_base="http://172.26.1.16:31251/v1",
                               api_key="sk-",
                               model_name="modelscope/deepseek-ai/deepseek-coder-33b-instruct")

    answer = llm_api.call_api_prompt(prompt=prompt, temperature=args.temperature)
    return answer



if __name__ == '__main__':
    llm_api = LocalCallAPI(api_base="http://172.26.1.16:31251/v1",
                          api_key="sk-",
                          model_name="modelscope/modelscope/Llama-2-70b-chat-ms")
    answer = llm_api.call_api(prompt_file='../template/EasySAT/bump_var_function/original_prompt.txt')
    print(answer)