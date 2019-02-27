#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import datetime
import subprocess
from subprocess import PIPE, CalledProcessError, check_call, Popen
from utils.app_utils import NonBlockingStreamReader as NBSR

cwd = os.getcwd()
log_directory = os.path.join(cwd,'logs')
log_file = os.path.join(log_directory,'error_logs.txt')
with open(log_file, "a+") as f:
    while True:
        try:
            process = Popen(['python','oddl.py'], stdout=PIPE, stderr=PIPE)
            poll = process.poll()
            #ret = process.returncode
            nbsr = NBSR(process.stdout)
            while poll == None:
                stdout = nbsr.readline(0.1)
                if not stdout:
                    break
                print(stdout.decode("utf-8"))
            else:
                print('Error {} with code {}'.format(stderr,ret))
        except Exception as e:
            f.write('{}. Failure to execute. Error:\t{}\n'.format(datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),str(e)))
            pass
