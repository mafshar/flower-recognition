import glob
import os
import sys
from shutil import copyfile
import scipy.io as sio

def generate_data_split(input_path, output_path):
    for root, dirs, files in os.walk(input_path):
        label = 0
        count = 0
        label_dir = os.path.join(output_path, str(label))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        for file in files:
            if file.endswith('.jpg'):
                if count > 79:
                    count = 0
                    label += 1
                    label_dir = os.path.join(output_path, str(label))
                    if not os.path.exists(label_dir):
                        os.makedirs(label_dir)
                src_filepath = os.path.join(input_path, file)
                dst_filepath = os.path.join(label_dir, file)
                copyfile(src_filepath, dst_filepath)
                count += 1
