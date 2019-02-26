#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import datetime
import subprocess
from subprocess import PIPE, CalledProcessError, check_call, Popen
import time

cwd = os.getcwd()
log_directory = os.path.join(cwd,'logs')
log_file = os.path.join(log_directory,'error_logs.txt')
with open(log_file, "a+") as f:
    while True:
        try:
            program = Popen(['python','oddl.py'], stdout=PIPE, stderr=PIPE)
            stdout, stderr = program.communicate()
            ret = program.returncode
            if ret:
                print('Error {} with code {}'.format(stderr,ret))
        except Exception as e:
            f.write('{}. Failure to execute. Error:\t{}'.format(datetime.datetime.now.strftime("%d/%m/%Y, %H:%M:%S"),str(e)))
            exit(0)
            pass
