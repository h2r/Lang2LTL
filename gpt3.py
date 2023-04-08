import os
from time import sleep
import logging
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("ORG_ID")


class GPT3:
    def __init__(self, engine, temp=0, max_tokens=128, n=1):
        self.engine = engine
        self.temp = temp
        self.max_tokens = max_tokens
        self.n = n

    def extract_ne(self, query, prompt=""):
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

    def generate(self, query_prompt):  # engines must match when compare two embeddings
        complete = False
        ntries = 0
        while not complete:
            try:
                raw_responses = openai.Completion.create(
                    model=self.engine,
                    prompt=query_prompt,
                    temperature=self.temp,
                    max_tokens=self.max_tokens,
                    stop=['\n'],
                    n=self.n,
                    # logprobs=5
                )
                complete = True
            except:
                sleep(30)
                logging.info(f"{ntries}: waiting for the server. sleep for 30 sec...\n{query_prompt}")
                logging.info('OK continue')
                ntries += 1
        if self.n == 1:
            responses = [raw_responses['choices'][0]['text'].strip()]
        else:
            responses = [choice['text'].strip() for choice in raw_responses['choices']]
        return responses

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def get_embedding(self, text):
        text = text.replace("\n", " ")  # replace newlines, which can negatively affect performance
        embedding = openai.Embedding.create(
            input=[text],
            model=self.engine  # change for different embedding dimension
        )["data"][0]["embedding"]
        return embedding


if __name__ == "__main__":
    gpt3 = GPT3("text-davinci-003", n=3)
    query_prompt = \
        "English: Go to Bookstore then to Science Library\n" \
        "Landmarks: Bookstore | Science Library\n" \
        "LTL: F ( Bookstore & F ( Science Library ) )\n\n" \
        "English: Go to Bookstore then reach Science Library\n" \
        "Landmarks: Bookstore | Science Library\n" \
        "LTL: F ( Bookstore & F ( Science Library ) )\n\n" \
        "English: Find Bookstore then go to Science Library\n" \
        "Landmarks: Bookstore | Science Library\n" \
        "LTL: F ( Bookstore & F ( Science Library ) )\n\n" \
        "English: Go to Burger Queen then to black stone park, but after KFC\n" \
        "Landmarks: Burger Queen | black stone park | KFC\n" \
        "LTL: F ( Burger Queen & F ( KFC & F ( black stone park ) )\n\n" \
        "English: Go to Burger Queen then to black stone park; go to KFC before black stone park and after Burger Queen\n"\
        "Landmarks: Burger Queen | black stone park | KFC\n" \
        "LTL: F ( Burger Queen & F ( KFC & F ( black stone park ) )\n\n" \
        "English: Go to Science Library then Burger Queen\n"
    response = gpt3.generate(query_prompt)
    print(response)

    gpt3 = GPT3("text-embedding-ada-002")
    embedding = gpt3.get_embedding("Burger Queen")
    # print(embedding)
    breakpoint()
