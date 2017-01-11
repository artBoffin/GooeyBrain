from flask import (Flask,jsonify, render_template, request)
import json

import os
import subprocess
import libs.GAN
# import libs.autoencoder as vae
import time

# webapp
app = Flask(__name__,static_folder='app/static', template_folder='app/templates')

# for debugging puerposes
app.config['TEMPLATES_AUTO_RELOAD'] = True

# training process and tensor board process are global to the app and can only run once at a time
global training_process
global tensor_board
training_process = None
tensor_board = None


@app.route('/api/dcgan', methods=['POST'])
def dcegan():
    global training_process
    global tensor_board
    arr = request.json
    print arr
    # flags = ["--%s %s"%(x['name'], x['value']) for x in arr]
    # cmd = ' '.join(['python', 'libs/run_tf.py']+flags)
    # training_process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # tensor_board = subprocess.Popen('tensorboard --logdir=logs',shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # print "started train and tensor board"
    return jsonify({'results':"trainig now!"})


@app.route('/api/is_training')
def is_training():
    global training_process
    global tensor_board
    training = False
    tensorboard = False
    out = " "
    if training_process is not None:
        #out = training_process.stdout.readline()
        training = True
    if tensor_board is not None:
        tensorboard = True
    return jsonify({"training":training,"tensorboard":tensorboard, "out":out})


@app.route('/api/get_def_parameters')
def get_parameters():
    params = dict()
    params['GAN'] = []
    for p in libs.GAN.GAN.parameters:
        params['GAN'].append(p.getJson())
    return jsonify(params)


    global training_process
    global tensor_board
    training = False
    tensorboard = False
    out = " "
    if training_process is not None:
        #out = training_process.stdout.readline()
        training = True
    if tensor_board is not None:
        tensorboard = True
    return jsonify({"training":training,"tensorboard":tensorboard, "out":out})

# @app.route('/api/vaegan', methods=['POST'])
# def vaegan():
#     arr = request.json
#     print request.json
#
#     p = {x['name']: x['value'] for x in arr}
#     vae.train_vaegan(p['files'],
#                          learning_rate=p['learning_rate'],
#                          batch_size=p['batch_size'],
#                          n_epochs=p['n_epochs'],
#                          n_examples=p['n_examples'],
#                          input_shape=p['input_shape'],
#                          crop_shape=p['crop_shape'],
#                          crop_factor=p['crop_factor'],
#                          n_filters=p['n_filters'],
#                          n_hidden=p['n_hidden'],
#                          n_code=p['n_code'],
#                          convolutional=p['convolutional'],
#                          variational=p['variational'],
#                          filter_sizes=p['filter_sizes'],
#                          activation=p['activation'],
#                          ckpt_name=p['ckpt_name'])
#     return jsonify(results="trainig now!")

@app.route('/api/save_parameters', methods=['POST'])
def save():
    parameters = request.json
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = "parameters-%s.json"%timestr

    directory = "saved_parameters"
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename)
    with open (filepath, 'w') as f:
        json.dumps(parameters, f)
    return jsonify("saved parameters to %s"%filepath)


@app.route('/')
def index():
    return render_template('index.html')

# models - GAN (for DCGAN set convolutional=true, VAE, VAEGAN, DRAW?)

    #for running tensor board:
#     call('tensorboard --logdir=logs')
