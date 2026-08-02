def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    freq = {}
    for sentence in sentences:
        for word in sentence:
            freq[word] = freq.get(word, 0) + 1
    return freq
