from os import path
import json

import fasttext

BEGINNING = 'B-'
INSIDE = 'I-'

class Dataset():
    def __init__(self, templates, literals):
        self.templates = templates
        self.literals = literals
        self.X = []
        self.Y = []

        self._prepare_data()
        
    def _prepare_data(self):
        # arbitary limit because the system hasn't needed more data
        for i in range(0, 20000):
            for template in self.templates:
                training_sample, tags_for_word = self._create_sample(template)

                self.X.append(training_sample)
                self.Y.append(tags_for_word)

        print(self.X[:10], self.Y[:10])

    def _create_sample(self, template):
        output = []
        current_template = template
        template_vector = []
        words = fasttext.tokenize(current_template)

        num_vars = current_template.count('{')
        count = 0

        tags_for_word = []
        training_sample = []
        for word in words:
            word = word.lower()
            if '{' in word:
                count += 1
                words_to_add, labels_to_add = self._process_variable(word)

                training_sample += words_to_add
                tags_for_word += labels_to_add
            else:
                if count == 0 or count >= num_vars:
                    tags_for_word.append('O')
                else:
                    tags_for_word.append('CC')
                training_sample.append(word)

        return training_sample, tags_for_word

    def _process_variable(self, word):
        labels = []
        words = []

        variable_word = word.replace('{', '').replace('}', '')
        try:
            words = fasttext.tokenize(self.literals[variable_word].get().lower())
        except:
            print(word, variable_word)

        for i in range(len(words)):
            label = BEGINNING + variable_word if i == 0 else INSIDE + variable_word
            labels.append(label)

        return words, labels

    def save(self, name):
        data_path = path.join(path.dirname(path.dirname(path.realpath(__file__))), 'data', 'en', name)
        with open(data_path, 'w', encoding='utf-8') as fp:
            json.dump(self.X[:300], fp, ensure_ascii=False)
