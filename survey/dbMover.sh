#!/bin/bash
echo "starting background threads for commiting redis to sqlite"
python survey/backgroundworker.py dbMover
