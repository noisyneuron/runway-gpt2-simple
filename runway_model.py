
import runway
import os
from runway.data_types import number, text, image, file, array
import random
import gpt_2_simple as gpt2
import re

setup_options = {
    'checkpoint': file(is_directory=True)
}

@runway.setup(options=setup_options)
def setup(opts):
    msg = '[SETUP] Ran with options: file = {}'
    print(msg.format(opts['checkpoint']))

    model = {
        'sess' : gpt2.start_tf_sess(),
        # 'modelname' : 'QBIGrun1-13700'
        'modelname' : opts['checkpoint']
    }

    gpt2.load_gpt2(model['sess'], model_name=model['modelname'], model_dir='')
    return model



generate_options = {
    'question': text(),
    'temperature' : number(min=0.0, max=1.0, step=0.01, default=0.8, description='the hotter the crazier'),
    'length' : number(min=1, max=1024, step=1, default=200, description='output char length')
}

@runway.command(name = 'generate', inputs = generate_options, outputs = { 'answer': text() })
def generate(model, args):

    print('[GENERATE] Ran with caption value "{}"'.format(args['question']))

    question = args['question']+"\n[QASEP]\n"

    txt = gpt2.generate(model['sess'],
          model_dir='',
          model_name=model['modelname'],
          prefix=question,
          length=args['length'],
          temperature=args['temperature'],
          top_p=0.9,
          top_k=40,
          nsamples=1,
          batch_size=1,
          return_as_list=True
          )

    print(txt)

    answer = txt[0]
    cleanedAns = 'I am too tired to answer right now. Go away.'

    if answer.startswith(question):
        answer = answer[len(question):]
        endIndex = answer.find("<|endoftext|>")
        if endIndex != -1:
            cleanedAns = answer[:endIndex]
        else:
            cleaned = re.findall(r".+[\.|\?]", answer, re.DOTALL)
            cleanedAns = cleaned[0]

    return cleanedAns



if __name__ == '__main__':
    runway.run(host='0.0.0.0', port=8000)
