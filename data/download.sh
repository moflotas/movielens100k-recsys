#!/bin/bash
wget -N -P raw "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
unzip -n raw/ml-100k.zip -d raw