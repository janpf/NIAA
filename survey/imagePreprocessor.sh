#!/bin/bash
echo "editing images on redis"
python survey/backgroundworker.py imagePreprocessor
