this is a simple unet, all parameters to run and tune it are in params.py. I like to make everything an absoute path but its up to you. 

everything should work, its maybe 2% ai generated to no worries there, i did barrow the nyu dataloader tho

added the slurm file(turing.sh) for turing, idk if its changed since i made the file also copying my original repo where i copyied most of my code from https://github.com/HudsonKortus/RBE474X its under p2

ai generated the requirements.txt and didnt test it so good luck:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


python3 train.py