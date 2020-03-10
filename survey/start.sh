#!/bin/bash
echo "creating/resetting database"
python survey/pre-start.py
echo "starting webserver"
gunicorn -w 4 -b 0.0.0.0:5000 --access-logfile "-" "survey.webserver:load_app()"
