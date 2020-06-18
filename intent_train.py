# -*- coding: utf-8 -*-
import json
from os import path
import io
import fasttext
import numpy
import json
from keras.preprocessing.sequence import pad_sequences

from literals_interface import LiteralsInterface, get_literals
from intent_trainer.trainer import Trainer
from intent_trainer.datautils import Vocabulary, Intent
from intent_trainer.preprocessing import Preprocessor
from intent_trainer.models import BiLSTMCRF
from intent_trainer.dataset import Dataset


BEGINNING = 'B-'
INSIDE = 'I-'


def train(schema, lang='en', root_dir=None):
    """ Gets the schema for the new intent passed in and parses it """
    intent = Intent(schema)
    # If no root_dir is passed in use the location of this file
    root_dir = root_dir if root_dir else path.dirname(path.realpath(__file__))

    config_path = path.join(path.dirname(path.realpath(__file__)), root_dir, "config")
    literals = get_literals(intent.literals)
    dataset = Dataset(intent.templates, literals)
    X, y = dataset.X, dataset.Y

    labels = Vocabulary(UnknownToken='O')
    labels.add('O')
    labels.add('CC')
    for variable in intent.variables:
        labels.add(BEGINNING + variable)
        labels.add(INSIDE + variable)

    #word embedding
    word_vocab = Vocabulary(UnknownToken='<unk>')
    word_vocab.add('<pad>')
    word_vocab.add('<unk>')
    word_vocab.build_vocab([w for command in X for w in command])

    #char embedding
    char_vocab = Vocabulary(UnknownToken='<unk>')
    char_vocab.add('<pad>')
    char_vocab.add('<unk>')
    char_vocab.build_vocab([ch for w in word_vocab for ch in w])

    #labels2idx, idx2label = labels.get_mappings()
    batch_size = 64
    epochs = 5

    if intent.variables:
        preprocessor = Preprocessor(WordVocab=word_vocab, LabelVocab=labels, CharVocab=char_vocab)
        model = BiLSTMCRF(labels, len(word_vocab), len(char_vocab))
        trainer = Trainer(model, X, y, preprocessor, split=[0.75, 0.95])

        trainer.train(batch_size, epochs)
        trainer.evaluate(labels.get_inverse_map())

        model.save_weights(path.join(config_path, 'weights', '%s.hdf5' % intent.name))
    
    dataset.save('%s.json' % intent.name)
    intent.save(path.join(config_path, "schemas", "%s_schema.json" % intent.name))
    labels.save(path.join(config_path, "labels", "%s_labels.json" % intent.name))
    word_vocab.save(path.join(config_path, "vocab", "%s_word_vocab.json" % intent.name))
    char_vocab.save(path.join(config_path, "vocab", "%s_char_vocab.json" % intent.name))
