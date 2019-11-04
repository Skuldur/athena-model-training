# Athena Intent Trainer

The Intent Trainer is responsible for creating new intent models and metadata based on a given intent schema. Additionally, it can create a base lambda for your intent that you can then use to deploy your intent to AWS.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

To create a new model you need to create an intent schema. `intents/en/config/schemas` contains examples of schemas. Below is a base example of an intent schema

**NB: The name of the schema file should always be `<name>_intent_schema.json` **

```json
{
    "name": "play",
    "templates": [
        "play {song} by {artist}",
        "play {song}",
        "play some {playlist}",
        "play music from {playlist}",
        "play something from {playlist}",
        "play something by {artist}",
        "play music by {artist}"
    ], 
    "variables": ["song", "artist", "playlist"],
    "literals": {
        "song": "ATHENA.SONG",
        "artist": "ATHENA.ARTIST",
        "playlist": "ATHENA.PLAYLIST"
    }
}
```

* **Name** refers to the name of the schema. The name of the files generated for this intent will be `<name>_intent_XXXX` 
* **Templates** is a list of the commands that this intent will handle. Make sure they are specific as if they clash with another intents command then it may cause some confusion for the classifier. To mark which parts of the command will be variables we use `{variable_name}`. In the above example we have three variables: song, artist, and playlist.
* **Variables** contains a list of all the variables in the schema **NOTE THIS IS LIKELY TO BE DEPRECATED IN A FUTURE VERSION**
* **Literals** tells the Intent Trainer which words should be used to replace the variable placeholders. The currently supported literals are:


| Literal Name       | Description          | Example                                    |
| ------------------ |:--------------------:| ------------------------------------------:|
| ATHENA.FOOD        | Names of food items  | Orange, Pizza, Almond Milk                 |
| ATHENA.SONG        | Names of songs       | Let it be, Shake It Off, Bohemian Rhapsody |
| ATHENA.ARTIST      | Names of artists     | The Beatles, Taylor Swift, Queen           |
| ATHENA.PLAYLIST    | Names of playlists   | Discover Weekly, Relaxation, Disney Hits   |
| ATHENA.PERCENT     | Percentage           | 1%, 10%, 53%                               |
| ATHENA.WORD_NUMBER | Numbers in word form | One, Fifty-Seven, One Hundred              |
| ATHENA.LANGUAGES   | Names of languages   | English, Spanish, Icelandic                |
| ATHENA.CITIES      | Names of cities      | Berlin, Miami, San Francisco               |

Currently we only support using the Athena literal. Future versions will support adding a list of values to the schema instead of the literal name.

### No Lambdas

```bash
python intent_trainer.py <path-to-schema>
```

However you will have to manually update the Intent Classifier 

```bash
python intent_classifier.py
```

### Lambdas

If you're using AWS Lambdas to run your intents handlers you can use

```bash
python add_intent.py <path-to-schema>
```

This command also creates an updated Intent Classifier weights file that includes the new intent. These weights will have to be manually moved to the intent classifier lambda and then it has to be redeployed.

## Model



## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
