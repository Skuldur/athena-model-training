# -*- coding: utf-8 -*-
import json
from os import path
import io
import fastText
import numpy
import json
from keras.preprocessing.sequence import pad_sequences
from layers import CRF
from sequences import TrainSequence
from preprocessing import IndexTransformer, pad_nested_sequences
from model import BiLSTMCRF
from literals_interface import LiteralsInterface

def get_word_id(vocab, word):
    if word in vocab:
        return vocab[word]
    return vocab['<unk>']

def get_char_id(vocab, char):
    if char in vocab:
        return vocab[char]
    return vocab['<unk>']

BEGINNING = 'B-'
INSIDE = 'I-'

class IntentTrainer():

    def __init__(self, schema, lang='en', root_dir=None):
        """ Gets the schema for the new intent passed in and parses it """
        self.name = "%s_intent" % schema['name']
        self.variables = schema.get('variables')
        self.templates = schema['templates']
        self.literals = schema.get('literals')
        self.vocab = set()
        self.vocab.add("<unk>")
        self.lang = lang
        self.schema = schema

        # If no root_dir is passed in use the location of this file
        self.root_dir = root_dir if root_dir else path.dirname(path.realpath(__file__))

    def add_to_vocab(self, word):
        self.vocab.add(word)


    def create_labels(self):
        labels = ['other', 'connect']

        # Only add variable labels if the schema has variables
        if self.variables:
            for label in self.variables:
                labels.append(BEGINNING + label)
                labels.append(INSIDE + label)

        return labels

    def prepare_data(self, dataset):
        training_X = []
        training_Y = []

        # arbitary limit because the system hasn't needed more data
        for i in range(0, 30000):
            for template in self.templates:
                training_sample, tags_for_word = self.create_sample(template, dataset)

                training_X.append(training_sample)
                training_Y.append(tags_for_word)

        return training_X, training_Y

    def create_sample(self, template, dataset):
        output = []
        current_template = template
        template_vector = []
        words = fastText.tokenize(current_template)

        num_vars = current_template.count('{')
        count = 0

        tags_for_word = []
        training_sample = []
        for word in words:
            if '{' in word:
                count += 1
                variable_word = word.replace('{', '').replace('}', '')
                try:
                    variable_target_list = fastText.tokenize(dataset[variable_word].get())
                except:
                    print(word, variable_word, words)

                tags_for_word.append(BEGINNING + variable_word)
                training_sample.append(variable_target_list[0].lower())
                self.add_to_vocab(variable_target_list[0].lower())

                for index in range(1, len(variable_target_list)):
                    self.add_to_vocab(variable_target_list[index].lower())
                    tags_for_word.append(INSIDE + variable_word)
                    training_sample.append(variable_target_list[index].lower())
            else:
                self.add_to_vocab(word.lower())
                if count == 0 or count >= num_vars:
                    tags_for_word.append('other')
                else:
                    tags_for_word.append('connect')
                training_sample.append(word.lower())

        return training_sample, tags_for_word

    def get_literals(self):
        # If the schema doesn't have variables, don't fetch literals
        if not self.literals:
            return {}

        dataset = {}
        for key, val in self.literals.items():
            dataset[key] = LiteralsInterface(val)

        return dataset

    def create_data(self):
        pass

    def train(self):
        data = self.get_literals()

        labels = self.create_labels()

        X, Y = self.prepare_data(data)

        n_words = len(self.vocab)
        print('words', n_words)

        vocab = sorted(list(self.vocab))

        # Word embedding
        vocab_map = dict((word, number) for number, word in enumerate(vocab))

        # Char embedding
        chars = sorted(list(set([w_i for w in vocab for w_i in w])))
        n_chars = len(chars)

        # Add two special tokens that represent words not in the vocabulary and padding
        char2idx = {c: i + 2 for i, c in enumerate(chars)}
        char2idx["<unk>"] = 1
        char2idx["<pad>"] = 0


        # Save the metadata for the Intent Model
        data_path = path.join(path.dirname(path.realpath(__file__)), 'data', self.lang, '%s.json' % self.name)
        with open(data_path, 'w', encoding='utf-8') as fp:
            json.dump(X[:300], fp, ensure_ascii=False)

        label_path = path.join(self.root_dir, "config", "labels", "%s_labels.json" % self.name)
        with open(label_path, 'w', encoding='utf-8') as fp:
            json.dump(labels, fp, ensure_ascii=False)

        word_path = path.join(self.root_dir, "config", "vocab", "%s_word_vocab.json" % self.name)
        with open(word_path, 'w', encoding='utf-8') as fp:
            json.dump(vocab_map, fp, ensure_ascii=False)

        char_path = path.join(self.root_dir, "config", "vocab", "%s_char_vocab.json" % self.name)
        with open(char_path, 'w', encoding='utf-8') as fp:
            json.dump(char2idx, fp, ensure_ascii=False)

        schema_path = path.join(self.root_dir, "config", "schema", "%s_intent_schema.json" % self.name)
        with open(schema_path, 'w', encoding='utf-8') as fp:
            json.dump(self.schema, fp, ensure_ascii=False)

        # The command has variables so we need to train a PoS tagger
        if self.variables:

            x_train, x_test, x_val = numpy.split(X, [int(len(X)*0.75), int(len(X)*0.95)])
            y_train, y_test, y_val = numpy.split(Y, [int(len(X)*0.75), int(len(X)*0.95)])
            batch_size = 64

            preprocessor = IndexTransformer()
            preprocessor.fit_with_char(vocab_map, labels, char2idx)

            train_seq = TrainSequence(x_train, y_train, batch_size, preprocessor.transform_with_char)
            test_seq = TrainSequence(x_test, y_test, batch_size, preprocessor.transform_with_char)

            model = BiLSTMCRF(labels, n_words, n_chars)
            model.build()
            model.compile()
            model.train(train_seq, test_seq)

            # Save the weights
            weights_path = path.join(self.root_dir, "config", "weights", '%s.hdf5' % self.name)
            model.model.save_weights(weights_path)

            vocab_map = dict((word, number) for number, word in enumerate(vocab))

            idx2label = dict((number, label) for number, label in enumerate(labels))

            sentences = x_val

            wrong = 0

            for sentence, true_labels in zip(sentences, y_val):
                words = [w for w in sentence]
                word_id_array = [[get_word_id(vocab_map, w) for w in sentence]]
                word_id_array = pad_sequences(sequences=word_id_array, padding="post", value=n_words)
                
                char_ids = [[[get_char_id(char2idx, ch) for ch in w] for w in sentence]]
                char_ids = pad_nested_sequences(char_ids)
                p = model.predict([numpy.array(word_id_array), numpy.array(char_ids)])

                predicted_labels = []
                for pred in p[0]:
                    predicted_labels.append(idx2label[pred])

                if predicted_labels != true_labels:
                    wrong += 1

            # Currently only using Accuracy
            # TODO: Add AUC ROC or at least Recall/Precision
            percentage = 100*(1.0*(len(sentences)-wrong) / len(sentences))
            print("Validation accuracy is %s percent" % percentage)




    