#!/bin/bash
echo "committing redis to sqlite..."
python survey/backgroundworker.py dbMover
