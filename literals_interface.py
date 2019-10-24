# -*- coding: utf-8 -*-
from os import path
import json

LITERALS_MAP = {
    'ATHENA.FOOD': {
        'file': 'food.literal',
        'sub': ''
    },
    'ATHENA.SONG': {
        'file': 'songs.literal',
        'sub': 'song'
    },
    'ATHENA.ARTIST': {
        'file': 'songs.literal',
        'sub': 'artist'
    },
    'ATHENA.PLAYLIST': {
        'file': 'songs.literal',
        'sub': 'playlist'
    },
    'ATHENA.PERCENT': {
        'file': 'percentage.literal',
        'sub': ''
    },
    'ATHENA.WORD_NUMBER': {
        'file': 'word_numbers.literal',
        'sub': ''
    },
    'ATHENA.LANGUAGES': {
        'file': 'languages.literal',
        'sub': ''
    },
    'ATHENA.CITIES': {
        'file': 'cities.literal',
        'sub': ''
    }
}

cwd_path = path.dirname(path.realpath(__file__))

class LiteralsInterface():
    def __init__(self, literal_name):
        literal_item = LITERALS_MAP[literal_name]

        literal_path = path.join(cwd_path, 'literals', literal_item['file'])
        with open(literal_path) as f:
            file_contents = json.load(f)

            if literal_item['sub']:
                self.items = [item[literal_item['sub']] for item in file_contents]
            else:
                self.items = file_contents


            self.length = len(self.items)
            self.iterator = 0

    def get_all(self):
        pass

    def get(self):
        value = self.items[self.iterator]
        self.iterator = (self.iterator + 1) % self.length
        return value

    def __len__(self):
        return self.length

    def set(self):
        pass
