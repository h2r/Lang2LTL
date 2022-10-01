import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


class GPT3:
    def __init__(self, temp=0.6):
        self.temp = temp

    def generate(self, query_prompt):
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=query_prompt,
            temperature=self.temp,
        )
        return response.choices[0].text

    def extract(self, query_prompt):
        out = self.generate(query_prompt)
        name_entities = out.split(' | ')

        placeholder_map = {}
        placeholder = "a"
        for idx, ne in enumerate(name_entities):
            placeholder_map[ne] = chr(ord(placeholder) + idx)  # increment placeholder using its ascii value

        return placeholder_map

    def translate(self, query_prompt):
        out = self.generate(query_prompt)
        return out


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

    gpt3.generate(query_prompt)
