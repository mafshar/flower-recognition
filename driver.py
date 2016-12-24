#!/usr/bin/env python

from data_reader import generate_data_split, scale_data


if __name__ == '__main__':
    generate_data_split('./data/images', './data')
    scale_data('./data/', 450, 450)
