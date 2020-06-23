
import runway
import gpt_2_simple as gpt2
from runway.data_types import number, text, image
from example_model import ExampleModel


setup_options = {
    'checkpoint': file(),
    # 'truncation': number(min=1, max=10, step=1, default=5, description='Example input.'),
    # 'seed': number(min=0, max=1000000, description='A seed used to initialize the model.')
}



@runway.setup(options=setup_options)
def setup(opts):
    msg = '[SETUP] Ran with options: seed = {}, truncation = {}'
    print(msg.format(opts['seed'], opts['truncation']))
    model = ExampleModel(opts)
    return model



@runway.command(name='generate',
                inputs={ 'caption': text() },
                outputs={ 'image': image(width=512, height=512) },
                description='Generates a red square when the input text input is "red".')
def generate(model, args):
    print('[GENERATE] Ran with caption value "{}"'.format(args['caption']))
    output_image = model.run_on_input(args['caption'])
    return {
        'image': output_image
    }



if __name__ == '__main__':
    runway.run(host='0.0.0.0', port=8000, debug=True)
