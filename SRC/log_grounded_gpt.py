import requests
from create_grounded_samples import *
from utils import *
import sys
import json
import os
import operator
import numpy as np

# gpt-2 imports
import fire
import tensorflow as tf
from azureml.core.run import Run
import model, sample, encoder
run = Run.get_context()

# python gpt-2/log_grounded_gpt.py gpt3 train 0 3p-3s


random.seed(30)

def get_sys_args():
    model_idx, split, start_idx, data = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    return model_idx, split, start_idx, data

def get_spatial_prompts(split="train", model="gpt"):
    if model == "gpt":
        max_len = 1024
    else:
        max_len = 400
    prompt = spatial_prompts(max_len=max_len)
    name = "spatial-0"
    return prompt, name


def get_grounded_words(data="gw"):

    print(data)
    if data == "gw":
        dirpath = "../data/grounded-words"
        files = os.listdir(dirpath)

    elif data == "only-rgb":
        dirpath = "../data/grounded-prompts/"
        files = ["colour-rgbs.tsv"]
    elif data == "only-name":
        dirpath = "../data/grounded-prompts/"
        files = [f for f in os.listdir(dirpath) if "colour-names" in f]
    elif data == "only-hex":
        dirpath = "../data/grounded-prompts/"
        files = [f for f in os.listdir(dirpath) if "colour-hexes" in f]

    words = []
    for file in files:
        print(file)
        f = open(os.path.join(dirpath, file), "r")
        lines = [line.strip() for line in f.readlines()]
        words.append(lines)

    return files, words

def get_dirpaths(type, model_idx, counter):
    prompt_dirpath = "../data/grounded-prompts/colour-prompts/%s/prompts/%s" % (type, str(counter))

    if os.path.isdir("../outputs/%s" % type) is False:
        os.mkdir(os.path.join("../outputs/", type))
    op_dirpath = os.path.join("../outputs/", type, model_idx)
    if os.path.isdir(op_dirpath) is False:
        os.mkdir(op_dirpath)

    op_dirpath = os.path.join("../outputs/", type, model_idx, str(counter))
    if os.path.isdir(op_dirpath) is False:
        os.mkdir(op_dirpath)

    return prompt_dirpath, op_dirpath


def log_gpt_outputs(seed=0):
    fire.Fire(interact_model)

def interact_model(
    model_name="124M",
    seed=42,
    nsamples=1,
    batch_size=1,
    length=300,
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir='models',
    print_op=False,
    type="3p-3s",
    counter=0,
):

    model_idx, split, start_idx, data = get_sys_args()
    type = data

    prompt_dirpath, op_dirpath = get_dirpaths(type, model_idx, counter)
    print(model_idx, data, counter)

    model_name = str(model_idx) + "M"
    seed = 42

    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    batch_size = 1

    print(nsamples, batch_size)
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)

    print(enc)
    print(model)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)



    files = os.listdir(prompt_dirpath)

    # get outputs
    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )


        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)


        for file in files:
            print("Logging " + str(file))

            f = open(os.path.join(prompt_dirpath, file), "r")
            lines = f.readlines()
            s = "\n".join(line.strip() for line in lines)
            f = open(os.path.join(op_dirpath, file), "w+")


            raw_text = s

            context_tokens = enc.encode(raw_text)

            # do this 3 times
            for _ in range(3):
                generated = 0
                for sample_idx in range(1):
                    out = sess.run(output, feed_dict={
                        context: [context_tokens for _ in range(batch_size)]
                    })[:, len(context_tokens):]

                    # batch size of 1
                    for i in range(1):
                        generated += 1
                        text = enc.decode(out[i])

                # take only the first line
                text = text.split("\n")[0]
                f.write(text + "\n")

def log_gpt(seed=0, type="3p-3s", counter=0):
    fire.Fire(interact_model)


if __name__=="__main__":
    model_idx, split, start_idx, data = get_sys_args()

    for counter in range(0, 6):

        if model_idx in ["124", "355", "774", "1558"]:
            log_gpt(counter=counter)

