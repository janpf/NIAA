#!/bin/bash
echo "commiting redis to sqlite..."
python survey/backgroundworker.py dbMover
