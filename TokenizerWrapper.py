from keras_preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np


class TokenizerWrapper(Tokenizer):

    def __init__(self, texts, padding, reverse=False, num_words=None):
        """
        :param texts: List of strings. This is the data-set.
        :param padding: Either 'post' or 'pre' padding.
        :param reverse: Boolean whether to reverse token-lists.
        :param num_words: Max number of words to use.
        """
        Tokenizer.__init__(self, num_words=num_words)
        self.fit_on_texts(texts)

        self.vocab_reverse = dict(zip(self.word_index.values(), self.word_index.keys()))

        # Convert all texts to lists of integer-tokens.
        # Sequences may have different lengths.
        self.tokens = self.texts_to_sequences(texts)

        # Very long sequences should be truncated at the end
        # Reversed sequences will be truncated at front so finally
        # they are actually truncated at the back
        if reverse:
            self.tokens = [list(reversed(x)) for x in self.tokens]
            truncating = 'pre'
        else:
            truncating = 'post'

        # length of each word sequence (tokenized)
        self.word_lengths = [len(x) for x in self.tokens]

        # Max number of tokens allowed -> (mean + 2*standard-deviation).
        self.max_tokens = np.mean(self.word_lengths) + 2 * np.std(self.word_lengths)
        self.max_tokens = int(self.max_tokens)

        # padding the sequences
        self.tokens_padded = pad_sequences(
            sequences=self.tokens,
            maxlen=self.max_tokens,
            padding=padding,
            truncating=truncating
        )

    def token_to_word(self, token):
        """
        :param token: token value for a particular word
        :return: Single word for the given @token
        """
        if token == 0:
            return " "
        return self.vocab_reverse.get(token, "")

    def tokens_to_text(self, tokens):
        text = [self.vocab_reverse.get(token, "") for token in tokens if token != 0]
        text = " ".join(text)
        return text

    def text_to_tokens(self, text, reverse=False, padding=False):
        """
        Convert a single text-string to tokens with optional
        reversal and padding.
        """
        tokens = self.texts_to_sequences(list(text))
        tokens = np.array(tokens)

        if reverse:
            tokens = np.flip(tokens, axis=1)
            truncating = 'pre'
        else:
            truncating = 'post'

        if padding:
            tokens = pad_sequences(tokens,
                                   maxlen=self.max_tokens,
                                   padding='pre',
                                   truncating=truncating)

        return tokens
