from chalice import Chalice
from chalicelib.controllers.intents import Intent


app = Chalice(app_name='athena-{intent_name}_intent-lambda')

intent = Intent()

@app.lambda_function()
def index(event, context):
    try:
        if not event:
            return "The request was empty"

        command = event['command']
        # No command was recorded
        if command is None or command == '':
            return 'No command in event'
        else:
            features = intent.parse(command)
            return intent.execute_command(features)
    except Exception as e:
        app.log.error(e)
        return str(e)
