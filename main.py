from flask import (Flask,jsonify, render_template, request, send_file)
import json

import os
import subprocess
import time
import libs.gan_model
import libs.dcgan_model
import signal
import logging

logging.basicConfig(level=logging.DEBUG)

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
    # save parameters to a temp txt files
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
    curr_dir = os.path.join(save_path, run_name)
    if not os.path.exists(curr_dir):
        os.makedirs(curr_dir)
    parameters_filename = "parameters%s" % time.strftime("%Y%m%d-%H%M%S")
    parameters_filepath = os.path.join(curr_dir,  parameters_filename)
    with open(parameters_filepath, 'w') as f:
        json.dump(parameters, f)
    return parameters_filepath, tensorboard_dir

@app.route('/api/train', methods=['POST'])
def dcegan():
    logging.info ("train called")
    global training_process
    global tensor_board
    arr = request.json

    parameters_filepath, tensorboard_dir = save_parameters_file(arr)

    # call the tensorflow train with the file as flag
    cmd = ' '.join(['python', 'run_tf.py', '--parameters_file', parameters_filepath])
    training_process = subprocess.Popen(
        cmd, shell=True) #, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    tensor_board = subprocess.Popen(
        'tensorboard --logdir=%s --host=%s --port=%s'%(tensorboard_dir, '127.0.0.1', '6006'),
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logging.info ("started tf train and tensor board")
    return jsonify({'results': "training now!"})


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
    return jsonify({"training": training, "tensorboard": tensorboard, "out": out})


@app.route('/api/get_def_parameters')
def get_parameters():
    params = dict()
    params['GAN'] = []
    for p in libs.dcgan_model.DCGAN.parameters:
        params['GAN'].append(p.getJson())
    return jsonify(params)

@app.route('/api/save_parameters', methods=['GET', 'POST'])
def save():
    parameters = request.json
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = "parameters-%s.json"%timestr
    directory = "./tmp/saved_parameters"
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename)
    with open (filepath, 'w') as f:
        json.dumps(parameters, f)
    return send_file(filepath, as_attachment=True)


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
  os.setpgrp() # create new process group, become its leader
  try:
      logging.info("starting Flask App")
      app.run(host='127.0.0.1', port=8000)
  finally:
      os.killpg(0, signal.SIGKILL) # kill all processes in my group

