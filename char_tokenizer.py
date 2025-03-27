class CharTokenizer:
    import torch as T
    
    """
    A simple character-level tokenizer that converts text to token IDs and vice versa.
    """
    def __init__(self, text=None, special_tokens=None):
        """
        Initialize the tokenizer, optionally building the vocabulary from input text.
        
        Args:
            text (str, optional): Text to build vocabulary from
            special_tokens (list, optional): List of special tokens to add to vocabulary
        """
        # Default special tokens
        self.special_tokens = {
            '<PAD>': 0,  # Padding token
            '<UNK>': 1,  # Unknown token
            '<BOS>': 2,  # Beginning of sequence
            '<EOS>': 3,  # End of sequence
        }
        
        # Update with custom special tokens if provided
        if special_tokens:
            for idx, token in enumerate(special_tokens):
                self.special_tokens[token] = idx
        
        # Initialize vocabulary dictionaries
        self.char_to_id = {}
        self.id_to_char = {}
        
        # Add special tokens to vocabulary
        for token, idx in self.special_tokens.items():
            self.char_to_id[token] = idx
            self.id_to_char[idx] = token
        
        # Build vocabulary if text is provided
        if text:
            self.build_vocab(text)
    
    def build_vocab(self, text):
        """
        Build vocabulary from text.
        
        Args:
            text (str): Text to build vocabulary from
        """
        # Get all unique characters from text
        unique_chars = sorted(set(text))
        
        # Assign IDs to characters (starting after special tokens)
        start_idx = len(self.special_tokens)
        for idx, char in enumerate(unique_chars):
            # Skip if character is already in vocabulary
            if char in self.char_to_id:
                continue
            
            char_idx = start_idx + idx
            self.char_to_id[char] = char_idx
            self.id_to_char[char_idx] = char
    
    def encode(self, text, add_special_tokens=False):
        """
        Convert text to token IDs.
        
        Args:
            text (str): Text to encode
            add_special_tokens (bool): Whether to add BOS/EOS tokens
            
        Returns:
            list: List of token IDs
        """
        token_ids = []
        
        # Add BOS token if requested
        if add_special_tokens:
            token_ids.append(self.special_tokens['<BOS>'])
        
        # Convert each character to its token ID
        for char in text:
            if char in self.char_to_id:
                token_ids.append(self.char_to_id[char])
            else:
                # Use UNK token for unknown characters
                token_ids.append(self.special_tokens['<UNK>'])
        
        # Add EOS token if requested
        if add_special_tokens:
            token_ids.append(self.special_tokens['<EOS>'])
        
        return token_ids
    
    def decode(self, token_ids, skip_special_tokens=True):
        """
        Convert token IDs back to text.
        
        Args:
            token_ids (list): List of token IDs
            skip_special_tokens (bool): Whether to skip special tokens in output
            
        Returns:
            str: Decoded text
        """
        # Filter special tokens if requested
        if skip_special_tokens:
            special_ids = list(self.special_tokens.values())
            token_ids = [id for id in token_ids if id not in special_ids]
        
        # Convert token IDs back to characters
        chars = []
        for id in token_ids:
            if id in self.id_to_char:
                chars.append(self.id_to_char[id])
            else:
                # This shouldn't happen unless there's an error
                chars.append(self.id_to_char[self.special_tokens['<UNK>']])
        
        return ''.join(chars)
    
    def save_vocab(self, path):
        """
        Save vocabulary to a file.
        
        Args:
            path (str): Path to save vocabulary
        """
        with open(path, 'w', encoding='utf-8') as f:
            for char, id in sorted(self.char_to_id.items(), key=lambda x: x[1]):
                f.write(f"{char}\t{id}\n")
    
    def load_vocab(self, path):
        """
        Load vocabulary from a file.
        
        Args:
            path (str): Path to vocabulary file
        """
        self.char_to_id = {}
        self.id_to_char = {}
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                char, id = line.strip().split('\t')
                id = int(id)
                self.char_to_id[char] = id
                self.id_to_char[id] = char
    
    def vocab_size(self):
        """
        Get vocabulary size.
        
        Returns:
            int: Vocabulary size
        """
        return len(self.char_to_id)


# Example usage
if __name__ == "__main__":
    # Sample text
    sample_text = "Hello, world! 你好，世界！"
    
    # Create tokenizer and build vocabulary
    tokenizer = CharTokenizer(sample_text)
    
    # Print vocabulary
    print(f"Vocabulary size: {tokenizer.vocab_size()}")
    print(f"Character to ID mapping: {tokenizer.char_to_id}")
    
    # Encode and decode
    encoded = tokenizer.encode(sample_text)
    print(f"Encoded: {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
    
    # With special tokens
    encoded_special = tokenizer.encode(sample_text, add_special_tokens=True)
    print(f"Encoded with special tokens: {encoded_special}")
    
    decoded_special = tokenizer.decode(encoded_special)
    print(f"Decoded (skipping special tokens): {decoded_special}") 