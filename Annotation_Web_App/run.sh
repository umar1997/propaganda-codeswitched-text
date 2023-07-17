ps aux | grep "flask run" | awk '{ print $2 }'|  xargs -n1 kill -9
export set FLASK_APP=app.py
export set FLASK_ENV=development
# FLASK_RUN_PORT
flask run

# <!-- https://flask.palletsprojects.com/en/1.1.x/config/ -->