import os

os.system('git clone https://github.com/openai/gpt-2.git')
os.system('wget https://raw.githubusercontent.com/tensorflow/tensor2tensor/master/tensor2tensor/utils/hparam.py')
os.system('mv hparam.py gpt-2/src')

# Migrating from tf 1.x to 2.x

def replacer(filename, toBeReplaced, replacementText):
    # Read in the file
    with open(filename, 'r') as file :
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace(toBeReplaced, replacementText)

    # Write the file out again
    with open(filename, 'w') as file:
        file.write(filedata)

filename = 'gpt-2/src/model.py'

toBeReplaced = '''
import tensorflow as tf
from tensorflow.contrib.training import HParams
'''
replacementText = '''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from hparam import HParams
'''

replacer(filename, toBeReplaced, replacementText)

toBeReplaced = 'import tensorflow as tf'
replacementText = '''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
'''

replacer('gpt-2/src/generate_unconditional_samples.py', toBeReplaced, replacementText)
replacer('gpt-2/src/interactive_conditional_samples.py', toBeReplaced, replacementText)
replacer('gpt-2/src/sample.py', toBeReplaced, replacementText)


# Install requirements
os.system('cd gpt-2')
os.system('cd gpt-2 && pip3 install -r requirements.txt')

# Download the models
os.system('cd gpt-2 && python3 download_model.py 124M')
os.system('cd gpt-2 && python3 download_model.py 355M')
os.system('cd gpt-2 && python3 download_model.py 774M')
os.system('cd gpt-2 && python3 download_model.py 1.5B')

# python3 src/interactive_conditional_samples.py
