

class GPT3:
    def __init__(self):
        return

    def generate(self, query_prompt):
        return ""

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
