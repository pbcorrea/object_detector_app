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
            process = Popen(['python','oddl.py'], stdout=PIPE, stderr=PIPE)
            stdout, stderr = process.communicate()
            poll = process.poll()
            ret = process.returncode
            print(stdout.decode("utf-8"))
            if poll != None:
                print('Error {} with code {}'.format(stderr,ret))
        except Exception as e:
            f.write('{}. Failure to execute. Error:\t{}\n'.format(datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),str(e)))
            pass
