#!/bin/bash

conda create -n p4ds python=3.9 -y
conda activate p4ds
conda install pip -y
pip install -r requirements.txt