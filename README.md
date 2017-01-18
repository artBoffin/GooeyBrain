# GooeyBrain

Gui app for generating images using [Tensorflow](https://www.tensorflow.org/)

### Installation
To run on your comuter:  
Clone the repository and cd to the dir
```
git clone https://github.com/artBoffin/GooeyBrain.git
cd GooeyBrain
```
Create a Python virtual enviroment and activate it
```
pip install virtualenv  
virtualenv venv  
source venv/bin/activate  
```
Install all Python requirments:
```
pip install -r requirements.txt  
```

Install required node modules:  
You need to have [node.js](https://nodejs.org) installed on your computer
```
npm install
```  

##### Run the app:
```
npm start
``` 

##### Running the app after installation  
In the app source dir type:
```
source venv/bin/activate 
npm start
```

##### Development
For development puerposes, you also want to run:
```
webpack --watch
```
In another shell window.To keep compiling React.js files

### Architecture
##### Frontend
`main.js` is the entry point that start an Electron application.  
All other frontend files are located in `app/`  
React.js files that are responsible for rendering all the components are in `app/src`
Webpack bundles everyuthing to `app/static/bundle.js`.  
For development, run `webpack --watch` in another shell window.  
   
Calling `npm start` is like calling `electron .`, which in turn starts the elcetron application.

##### Backend
The Electron application spawns a Python process starting `main.py`, which starts a Flask app.
`models_manager.py` is a help file for taking care of all different Deep Learning models, 
making it simple to add a new models (currently, only dcgan model exsists)  
`run_tf.py` is called as a subprocess from main.py when "Train" ot "Generate" buttons are clicked.
It is givena parameters filepath in it's arguments and a boolean if this is a Train or not.

#### To Add a New Model:
Add a folder with the model name in the "models" directory
Make sure to create an `__init__.py` file and define `get_model` and `get_parameters`
Create a class that extendes Base_model (include `train` and `generate` functions).

##TODO:
 - use stepper to encapsualte the process:
   - First the user selects if she wants to train a new model or load exsisting one
     - if trainin new, select what model from a list of exsisting models, only then parameters list is shown
     - else, uplaod a parameters file and a path to a Tensorflow chekcoint directory
 - "Generate" button
 - "Load Parameters" button
 - Add uplaod example image option, and use javascript to detemine input size
 - Fix the pullover jump when clicking the question marks
