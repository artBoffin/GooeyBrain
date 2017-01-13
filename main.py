from flask import (Flask,jsonify, render_template, request)
import json

import os
import subprocess
import time
import libs.gan_model

# webapp
app = Flask(__name__,static_folder='app/static', template_folder='app/templates')

# for debugging puerposes
app.config['TEMPLATES_AUTO_RELOAD'] = True

# training process and tensor board process are global to the app and can only run once at a time
global training_process
global tensor_board
training_process = None
tensor_board = None

def save_parameters_file(parameters):
    # save parameters to temp txt file
    save_path = "./tmp"
    run_name = ""
    tensorboard_dir = "./tmp/logs"
    for p in parameters:
        if p['name']=="save_path":
            save_path = p['value']
        if p['name']=="run_name":
            run_name = p['value']

    tensorboard_dir = os.path.join(save_path, run_name, 'logs')
    parameters.append({'name': 'tensorboard_dir', 'value':tensorboard_dir})
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    parameters_filename = "parameters%s" % time.strftime("%Y%m%d-%H%M%S")
    parameters_filepath = os.path.join(save_path, run_name,  parameters_filename)
    with open(parameters_filepath, 'w') as f:
        json.dump(parameters, f)
    return parameters_filepath, tensorboard_dir

@app.route('/api/train', methods=['POST'])
def dcegan():
    global training_process
    global tensor_board
    arr = request.json

    parameters_filepath, tensorboard_dir = save_parameters_file(arr)

    # call the tensorflow train with the file as flag
    cmd = ' '.join(['python', 'run_tf.py', '--parameters_file', parameters_filepath])
    training_process = subprocess.Popen(cmd, shell=True) #, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    tensor_board = subprocess.Popen('tensorboard --logdir=%s'%tensorboard_dir,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
    for p in libs.gan_model.GAN.parameters:
        params['GAN'].append(p.getJson())
    return jsonify(params)

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
