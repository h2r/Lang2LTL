import os
from time import sleep
import logging
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("ORG_ID")


class GPT3:
    def __init__(self, engine, temp=0, max_tokens=128, n=1, stop=['\n']):
        self.engine = engine
        self.temp = temp
        self.max_tokens = max_tokens
        self.n = n
        self.stop = stop

    def extract_re(self, query, prompt=""):
        query_prompt = prompt + query
        outs = self.generate(query_prompt)
        name_entities = outs[0].split(' | ')
        return name_entities

    def translate(self, query, prompt=""):
        if isinstance(query, list):
            query = query[0]
        query_prompt = prompt + query
        outs = self.generate(query_prompt)
        return outs

    def generate(self, query_prompt):
        complete = False
        ntries = 0
        while not complete:
            try:
                raw_responses = openai.Completion.create(
                    model=self.engine,
                    prompt=query_prompt,
                    temperature=self.temp,
                    max_tokens=self.max_tokens,
                    stop=self.stop,
                    n=self.n,
                    # logprobs=5
                )
                complete = True
            except:
                sleep(30)
                logging.info(f"{ntries}: waiting for the server. sleep for 30 sec...\n{query_prompt}")
                logging.info("OK continue")
                ntries += 1
        if self.n == 1:
            responses = [raw_responses["choices"][0]["text"].strip()]
        else:
            responses = [choice["text"].strip() for choice in raw_responses["choices"]]
        return responses

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def get_embedding(self, text):  # engines must match when compare two embeddings
        text = text.replace("\n", " ")  # replace newlines, which can negatively affect performance
        embedding = openai.Embedding.create(
            input=[text],
            model=self.engine  # change for different embedding dimension
        )["data"][0]["embedding"]
        return embedding


class GPT4:
    def __init__(self, engine="gpt-4", temp=0, max_tokens=128, n=1, stop=['\n']):
        self.engine = engine
        self.temp = temp
        self.max_tokens = max_tokens
        self.n = n
        self.stop = stop

    def extract_re(self, query, prompt=""):
        query_prompt = prompt + query
        outs = self.generate(query_prompt)
        name_entities = outs[0].split(' | ')
        return name_entities

    def translate(self, query, prompt=""):
        if isinstance(query, list):
            query = query[0]
        query_prompt = prompt + query
        outs = self.generate(query_prompt)
        return outs

    def generate(self, query_prompt):
        complete = False
        ntries = 0
        while not complete:
            try:
                raw_responses = openai.ChatCompletion.create(
                    model=self.engine,
                    messages=prompt2msg(query_prompt),
                    temperature=self.temp,
                    n=self.n,
                    stop=self.stop,
                    max_tokens=self.max_tokens,
                )
                complete = True
            except:
                sleep(30)
                logging.info(f"{ntries}: waiting for the server. sleep for 30 sec...\n{query_prompt}")
                logging.info("OK continue")
                ntries += 1
        if self.n == 1:
            responses = [raw_responses["choices"][0]["message"]["content"].strip()]
        else:
            responses = [choice["message"]["content"].strip() for choice in raw_responses["choices"]]
        return responses

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def get_embedding(self, text):  # engines must match when compare two embeddings
        text = text.replace("\n", " ")  # replace newlines, which can negatively affect performance
        embedding = openai.Embedding.create(
            input=[text],
            model=self.engine  # change for different embedding dimension
        )["data"][0]["embedding"]
        return embedding


def prompt2msg(query_prompt):
    """
    Make prompts for GPT-3 compatible with GPT-3.5 and GPT-4.
    Support prompts for
        RER: e.g., data/osm/rer_prompt_16.txt
        symbolic translation: e.g., data/prompt_symbolic_batch12_perm/prompt_nexamples1_symbolic_batch12_perm_ltl_formula_9_42_fold0.txt
        end-to-end translation: e.g., data/osm/osm_full_e2e_prompt_boston_0.txt
    :param query_prompt: prompt used by text completion API (text-davinci-003).
    :return: message used by chat completion API (gpt-3, gpt-3.5-turbo).
    """
    # prompt_splits = query_prompt.split("\n\n")
    # system_prompt = "\n\n".join(prompt_splits[0: -1])  # task description and common examples
    # query = prompt_splits[-1]  # specific context info and query question
    #
    # msg = [{"role": "system", "content": system_prompt}]
    # msg.append({"role": "user", "content": query})

    prompt_splits = query_prompt.split("\n\n")
    task_description = prompt_splits[0]
    examples = prompt_splits[1: -1]
    query = prompt_splits[-1]

    msg = [{"role": "system", "content": task_description}]
    for example in examples:
        if "\n" in example:
            example_splits = example.split("\n")
            q = '\n'.join(example_splits[0:-1])  # every line except the last in 1 example block
            a_splits = example_splits[-1].split(" ")  # last line is the response
            q += f"\n{a_splits.pop(0)}"
            a = " ".join(a_splits)
            msg.append({"role": "user", "content": q})
            msg.append({"role": "assistant", "content": a})
        else:  # info should be in system prompt, e.g., landmark list
            msg[0]["content"] += f"\n{example}"
    msg.append({"role": "user", "content": query})

    return msg


if __name__ == "__main__":
    # gpt3 = GPT3("text-davinci-003", n=3)
    # query_prompt = \
    #     "English: Go to Bookstore then to Science Library\n" \
    #     "Landmarks: Bookstore | Science Library\n" \
    #     "LTL: F ( Bookstore & F ( Science Library ) )\n\n" \
    #     "English: Go to Bookstore then reach Science Library\n" \
    #     "Landmarks: Bookstore | Science Library\n" \
    #     "LTL: F ( Bookstore & F ( Science Library ) )\n\n" \
    #     "English: Find Bookstore then go to Science Library\n" \
    #     "Landmarks: Bookstore | Science Library\n" \
    #     "LTL: F ( Bookstore & F ( Science Library ) )\n\n" \
    #     "English: Go to Burger Queen then to black stone park, but after KFC\n" \
    #     "Landmarks: Burger Queen | black stone park | KFC\n" \
    #     "LTL: F ( Burger Queen & F ( KFC & F ( black stone park ) )\n\n" \
    #     "English: Go to Burger Queen then to black stone park; go to KFC before black stone park and after Burger Queen\n"\
    #     "Landmarks: Burger Queen | black stone park | KFC\n" \
    #     "LTL: F ( Burger Queen & F ( KFC & F ( black stone park ) )\n\n" \
    #     "English: Go to Science Library then Burger Queen\n"
    # response = gpt3.generate(query_prompt)
    # print(response)

    # gpt3 = GPT3("text-embedding-ada-002")
    # embedding = gpt3.get_embedding("Burger Queen")
    # print(embedding)

    gpt4 = GPT4()
    query_prompt = \
        "Your tasks is to repeat exact strings from the given utterance which possibly refer to certain propositions." \
        "English: Go to Bookstore then to Science Library\n" \
        "Landmarks: Bookstore | Science Library\n\n" \
        "English: Go to Bookstore then reach Science Library\n" \
        "Landmarks: Bookstore | Science Library\n\n" \
        "English: Find Bookstore then go to Science Library\n" \
        "Landmarks: Bookstore | Science Library\n\n" \
        "English: Go to Burger Queen then to black stone park, but after KFC\n" \
        "Landmarks: Burger Queen | black stone park | KFC\n\n" \
        "English: Go to Burger Queen then to black stone park; go to KFC before black stone park and after Burger Queen\n" \
        "Landmarks: Burger Queen | black stone park | KFC\n\n" \
        "English: Go to Science Library then Burger Queen\n" \
        "Landmarks:"
    response = gpt4.generate(query_prompt)
    print(response)

    breakpoint()
