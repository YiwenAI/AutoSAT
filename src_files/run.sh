#!/bin/bash
service ssh start
sshfs -o StrictHostKeyChecking=no data_server@172.24.4.18:/data/data_server/tizero /root/data_server