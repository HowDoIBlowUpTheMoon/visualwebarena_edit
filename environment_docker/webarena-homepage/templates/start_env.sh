#!/bin/bash

#CURR_DIR=$PWD

cd /local/scratch/lin\.3976/web_eval/classifieds_docker_compose
bash /local/scratch/lin\.3976/web_eval/classifieds_docker_compose/docker_classifieds.sh

cd /local/scratch/lin\.3976/web_eval
bash /local/scratch/lin\.3976/web_eval/docker_shop.sh
bash /local/scratch/lin\.3976/web_eval/docker_social.sh
bash /local/scratch/lin\.3976/web_eval/docker_wiki.sh

cd /local/scratch/lin\.3976/web_eval/visualwebarena_edit/environment_docker/webarena-homepage
flask run --host=0.0.0.0 --port=4399

