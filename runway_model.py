
import runway
import os
from text_model import TextModel
from runway.data_types import number, text, image, file, array



setup_options = {
    'checkpoint': file(is_directory=True),
    # 'truncation': number(min=1, max=10, step=1, default=5, description='Example input.'),
    # 'seed': number(min=0, max=1000000, description='A seed used to initialize the model.')
}



@runway.setup(options=setup_options)
def setup(opts):
    msg = '[SETUP] Ran with options: file = {}'
    print(msg.format(opts['checkpoint']))
    model = TextModel(options)
    # model = TextModel({'checkpoint': 'QBIGrun1-13700'})
    return model



@runway.command(name='generate',
                inputs={ 'question': text() },
                outputs={ 'answer': array(text) },
                description='Generates a red square when the input text input is "red".')
def generate(model, args):
    print('[GENERATE] Ran with caption value "{}"'.format(args['question']))
    output = model.run_on_input(args['question'])
    return {
        'answer': output
    }



if __name__ == '__main__':
    runway.run(host='0.0.0.0', port=8000)
