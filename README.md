# Advanced Topics in Machine Learning

Here, we attempt to reproduce the results in the paper titled 'Mapping Language Models to Grounded Conceptual Spaces'.

## Running it on Google Colab

Try uploading 'setup_colab.ipynb' to Google Colab and attempt to run it. The script will automatically clone this repository and run the 'main.py' file.

## Setting up the key for GPT-3

Get a new key issued from the following URL.
https://beta.openai.com/account/api-keys

### For your local machine
Run the following commands to save the key in an environment variable.
'''
echo "export OPENAI_API_KEY='yourkey'" >> ~/.zshrc
source ~/.zshrc
'''

### For Google Colab
Copy paste the main file content into a cell. Update the value for the key manually.