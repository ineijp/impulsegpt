class CharTokenizer():
    def __init__(self, text):
        unique_chars = set(text)
        self.num_unique_chars = len(unique_chars)
        self.char_to_id = {char: i for i, char in enumerate(unique_chars)}
        self.id_to_char = {i: char for i, char in enumerate(unique_chars)}
    
    def encode(self, x):
        return [self.char_to_id[i] for i in x]
    
    def decode(self, x):
        return [self.id_to_char[i] for i in x]