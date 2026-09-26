import tiktoken

ENDOFTEXT = "<|endoftext|>"

DOCUMENTS = [
    "Hello, do you like tea.",
    "In the sunlit terraces of someunknownPlace.",
]


def main() -> None:
    encoding = tiktoken.get_encoding("gpt2")
    print("vocab size:", encoding.n_vocab)
    print("endoftext id:", encoding.eot_token)

    # One flat id sequence, documents separated by <|endoftext|>.
    text = ENDOFTEXT.join(DOCUMENTS)
    ids = encoding.encode(text, allowed_special={ENDOFTEXT})
    print("documents:", DOCUMENTS)
    print("encoded:", ids)
    print("decoded:", encoding.decode(ids))

    print("subword split:")
    for token_id in encoding.encode("someunknownPlace"):
        print(f"  {token_id}\t{encoding.decode([token_id])!r}")


if __name__ == "__main__":
    main()
