#!/bin/bash
echo "starting background threads for image editing"
python survey/backgroundworker.py imagePreprocessor
