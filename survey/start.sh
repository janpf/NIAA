#!/bin/bash
python survey/pre-start.py
gunicorn -w 12 -b 0.0.0.0:5000 survey.webserver:load_app()
