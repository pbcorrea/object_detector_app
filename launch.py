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
            process = Popen(['python','oddl.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
            stdout, stderr = process.communicate()
            #poll = process.poll()
            ret = process.returncode
            #while poll == None:
            #    try:
            #        print(stdout.decode("utf-8"))
            #    except:
            #        pass
            #if poll != None:
            print(stdout.decode("utf-8"))
            print('Error: {}. Exit with code {}'.format(stderr,ret))
        except Exception as e:
            f.write('{}. Failure to execute. Error:\t{}\n'.format(datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),str(e)))
            pass
