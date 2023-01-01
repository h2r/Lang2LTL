import os
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("ORG_ID")


class GPT3:
    def extract_ne(self, query, **kwargs):
        query_prompt = kwargs["prompt"] + query + "\nLandmarks:"
        out = GPT3.generate(query_prompt)
        name_entities = out.split(' | ')
        return name_entities

    def translate(self, query, **kwargs):
        query_prompt = kwargs["prompt"] + query + "\nLTL:"
        outs = GPT3.generate(query_prompt, n=kwargs["n"])
        if kwargs["n"] > 1:  # GPT-3 generates more than 1 candidate translations
            return [out.strip() for out in outs]
        return outs.strip()

    @staticmethod
    def generate(query_prompt, engine="text-davinci-003", temp=0.6, n=1):  # engines must match when compare two embeddings
        response = openai.Completion.create(
            model=engine,
            prompt=query_prompt,
            temperature=temp,
            n=n
        )['choices'][0]['text']
        return response

    @staticmethod
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def get_embedding(text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")  # replace newlines, which can negatively affect performance
        embedding = openai.Embedding.create(
            input=[text],
            model=model  # change for different embedding dimension
        )["data"][0]["embedding"]
        return embedding


if __name__ == '__main__':
    gpt3 = GPT3()

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
        "English: "

    response = GPT3.generate(query_prompt, n=3)
    print(response)
    # embedding = GPT3.get_embedding("Burger Queen")
    # print(embedding)
    breakpoint()
