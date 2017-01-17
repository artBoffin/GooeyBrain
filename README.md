# GooeyBrain



For running locally:

pip install virtualenv
virtualenv venv;
source venv/bin/activate
pip install -r requirements.txt
npm install -g webpack;
npm install


Running the app

If you're using a virtualenv, activate it.

source venv/bin/activate
Then run the Flask app:

gunicorn main:app