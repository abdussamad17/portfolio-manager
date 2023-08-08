#!/bin/bash
LOGFILE=/home/user/logs/`date -I`.log

bash -i /home/user/github/portfolio-manager/run.sh >> $LOGFILE 2>&1