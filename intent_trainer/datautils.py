from os import path
import math
import json

import numpy as np
from keras.utils import Sequence

class Vocabulary():
    def __init__(self, UnknownToken='<unk>'):
        self.vocab = {}
        self.unk = UnknownToken

    def add(self, item):
        if not item in self.vocab:
            self.vocab[item] = len(self.vocab)

    def build_vocab(self, items):
        for item in items:
            self.add(item)

    def get(self, item):
        """Gets the integer ID of the item if it exists in the vocabulary,
        Otherwise it gets the ID of the unknown token
        """
        return self.vocab.get(item, self.vocab.get(self.unk))

    def get_inverse_map(self):
        return {value: key for key, value in self.vocab.items()}

    def save(self, name):
        with open(name, 'w', encoding='utf-8') as fp:
            json.dump(self.vocab, fp, ensure_ascii=False)

    def load(self, name):
        with open(name, encoding="utf8") as f:
            self.vocab = json.load(f)
            self.length = len(self.vocab)

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item):
        return self.get(item)

    def __contains__(self, item):
        return item in self.vocab

    def __iter__(self):
        for w in self.vocab:
            yield w


class Intent():
    def __init__(self, schema=None):
        if schema:
            self._init_attributes(schema)

    def _init_attributes(self, schema):
        self.schema = schema
        self.name = "%s_intent" % schema['name']
        self.templates = schema['templates']
        self.literals = schema.get('literals', {})
        self.variables = [key for key in self.literals]

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as fp:
            json.dump(self.schema, fp, ensure_ascii=False)

    def load(self, path):
        with open(path, encoding="utf8") as f:
            schema = json.load(f)
            self._init_attributes(schema)


class DataSequence(Sequence):
    def __init__(self, x, y, batch_size=1, preprocess=None):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.preprocess = preprocess

    def __getitem__(self, idx):
        """Generates the next batch for the model and returns the preprocessed data
        """
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]

        return self.preprocess(batch_x, batch_y)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)