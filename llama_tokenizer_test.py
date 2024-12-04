from transformers import AutoTokenizer

MAX = 3

if __name__ == "__main__":
    # Load the LLaMA tokenizer (update with your specific model)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    print("Loaded tokenizer")

    # Tokens to check
    colons = []
    colon_str = ""
    for i in range(MAX+1):
        colons.append(colon_str)
        colon_str += ":"
    tokens_to_check = []
    for x in colons:
        for i in range(MAX+1):
            tokens_to_check.append(x)
            x += "\n"

    print(tokens_to_check)

    # Tokenize and retrieve IDs
    token_ids = []
    for token in tokens_to_check:
        ids = tokenizer.convert_tokens_to_ids(token)
        print(f"Token: '{token}' -> Token ID: {ids}")
        if ids is not None:
            token_ids.append(ids)

    deduped_tokens = list(set(token_ids))

    print(f"Unique token IDs: {deduped_tokens}")
