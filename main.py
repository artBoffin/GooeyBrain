from flask import Flask, jsonify, render_template, request, redirect, url_for
from subprocess import call

# webapp
app = Flask(__name__,static_folder='app/static', template_folder='app/templates')

# for debugging puerposes
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.route('/api/vaegan', methods=['POST'])
def vaegan():
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    output1 = regression(input)
    output2 = convolutional(input)
    return jsonify(results=[output1, output2])

@app.route('/api/dcgan', methods=['POST'])
def dcgan():
    print request.form
    print request.form.keys()
    return redirect(url_for('index'))
#     input = request.json
#     print request.json
#     return jsonify(results="trainig now!")


@app.route('/')
def index():
    return render_template('index.html')

    #for running tensor board:
#     call('tensorboard --logdir=logs')