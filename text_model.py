
import random
import gpt_2_simple as gpt2
from PIL import Image

class TextModel():

    def __init__(self, options):
        #random.seed(options['seed'])
        self.sess = gpt2.start_tf_sess()
        gpt2.load_gpt2(self.sess, model_name=options['checkpoint'])

    # Generate an image based on some text.
    def run_on_input(self, caption_text):

        txt = gpt2.generate(self.sess,
              model_name=model_name,
              prefix=caption_text,
              length=200,
              temperature=1.0,
              top_p=0.9,
              top_k=40,
              nsamples=5,
              batch_size=5,
              return_as_list=True
              )

        return txt
