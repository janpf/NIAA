#!/bin/bash
echo "creating/resetting database"
python survey/pre-start.py

echo "starting background threads for image editing"
python survey/image_edit_process.py &

echo "starting webserver"
gunicorn -w 4 -b 0.0.0.0:5000 --reload --access-logfile "-" "survey.webserver:load_app()"
