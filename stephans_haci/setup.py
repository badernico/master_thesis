#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# TODO: add license

import glob
from setuptools import setup, find_packages

setup(
    name         = 'haci',
    version      = open('VERSION').read().rstrip('\r\n'),
    author       = 'Stephan Lenk, Hartwig Deneke',
    author_email = 'lenk@tropos.de',
    scripts      =  glob.glob('bin/*'),
    package_dir  = { '': 'src'},
    packages     = [ 'haci' ],
    package_data = { 'haci': ['share/*']},
    zip_safe     = False
)

