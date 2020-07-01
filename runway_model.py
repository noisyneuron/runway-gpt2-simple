
import runway
import os
from runway.data_types import number, text, image, file, array
import random
import gpt_2_simple as gpt2


setup_options = {
    'checkpoint': file(is_directory=True)
}

@runway.setup(options=setup_options)
def setup(opts):

    msg = '[SETUP] Ran with options: file = {}'
    print(msg.format(opts['checkpoint']))
    model = {
        'sess' : gpt2.start_tf_sess(),
        'modelname' : opts['checkpoint']
    }
    # self.sess = gpt2.start_tf_sess()
    # self.modelname = opts['checkpoint']
    gpt2.load_gpt2(model['sess'], model_name=model['modelname'], model_dir='')

    # model = TextModel(opts)
    # model = TextModel({'checkpoint': 'QBIGrun1-13700'})

    return model


generate_options = {
    'question': text(),
    'temperature' : number(min=0.0, max=1.0, step=0.01, default=0.8, description='the hotter the crazier'),
    'length' : number(min=1, max=1024, step=1, default=200, description='output char length'),
    'samples' : number(min=1, max=5, step=1, default=1, description='number of answers to generate')
}

@runway.command(name = 'generate', inputs = generate_options, outputs = { 'answer': array(text) })
def generate(model, args):

    print('[GENERATE] Ran with caption value "{}"'.format(args['question']))

    txt = gpt2.generate(model['sess'],
          model_dir='',
          model_name=model['modelname'],
          prefix=question,
          length=inputs['length'],
          temperature=inputs['temperature'],
          top_p=0.9,
          top_k=40,
          nsamples=inputs['samples'],
          batch_size=inputs['samples'],
          return_as_list=True
          )

    # return txt
    # output = model.run_on_input(args['question'])

    return { 'answer': txt }



if __name__ == '__main__':
    runway.run(host='0.0.0.0', port=8000)
