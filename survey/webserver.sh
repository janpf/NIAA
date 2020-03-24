#!/bin/bash
echo "creating/resetting database"
python survey/pre-start.py

echo "starting webserver"
gunicorn -w 5 -b 0.0.0.0:5000 --reload --access-logfile "-" "survey.webserver:load_app()"
