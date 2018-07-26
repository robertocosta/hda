Links:
- https://gitlab.dei.unipd.it/costarob/HDACostaProspero/raw/master/jobs.txt
- https://gitlab.dei.unipd.it/costarob/HDACostaProspero/raw/master/job_table.txt
- https://gitlab.dei.unipd.it/costarob/HDACostaProspero/raw/master/mat/list.txt
- https://gitlab.dei.unipd.it/costarob/HDACostaProspero/raw/master/txt/list.txt

Windows settings:
- Anaconda prompt (as Admin)
- conda update conda
- conda list
- conda install msgpack-python
- conda install -c anaconda graphviz
- python -m pip install --upgrade pip
- python -m pip install msgpack
- pip3 install --upgrade tensorflow
- pip install keras
- pip install pydot
- python -m pip install pydot-ng
- python -m pip install graphviz
- //conda install keras
- //conda update --all
- //python -m pip install tensorflow
- //python -m pip install keras==2.0.7 tensorflow==1.2.1 pydot
- //pip install --ignore-installed --upgrade tensorflow 

Git settings:
- Create git account for gitlab.dei.unipd.it 
- Open Ubuntu (or Bash for windows) and set the globals
	- git config --global user.name "Roberto Costa"
	- git config --global user.email "costarob@dei.unipd.it"
- Clone the repository
	- cd WHEREVER
	- git clone https://gitlab.dei.unipd.it/costarob/HDACostaProspero.git
- Enter the folder and edit one file
	- cd HDACostaProspero
	- nano Models.py
- Add the changes
	- git add Models.py
- Commit the changes
	- git commit -m "description of the architectures"
- Upload
	- git push
- Execute the script
	- ssh prosperolo@login.dei.unipd.it /nfsd/hda/prosperolo/start_main.sh
- Execute the script for uploading to gitlab the results (when finished)
	- ssh prosperolo@login.dei.unipd.it /nfsd/hda/prosperolo/push.sh
- Display the results with MATLAB: plots.m

Reading outputs: file jobs.txt
- MonJul232333532018,1711896,https://gitlab.dei.unipd.it/costarob/HDACostaProspero/blob/9e4c1154466b036e4cfb1845cd12410f8c576fbd/Models.py 
- Format: Date,JobID,URL
    - Date: Monday, July, the 23rd, at 23:33:53, year 2018
    - Job ID: 1711896
    - URL: URL of the version of the code with the NN architecture
- file with the output: /nfsd/hda/prosperolo/HDACostaProspero/jmain.job.e1711896

Reading outputs with MATLAB
- execute m/plots.m
- click on the image displayed to go to the next
- images are saved in m/filename.png
    where filename is the concatenation of the following fields separated by the character '_' :
    - the type of architecture (FFNN / CNN)
    - the parameters:
        - the period, e.g. t-10 means 10 samples per block
        - the number of conv filters (in the FFNN this parameter is not used).
        e.g. nfilters-32 means that each conv layer has 32 filters
        - the number of neurons in the fully connected layers.
        e.g. ndense-256 means that there are 256 neurons in the fully connected layers
        - number of classes.
        e.g. ncl-6 means that 6 classes have been considered
        - batch size
        - number of epochs
        - learning rate
        - type of loss: either
            - Mean squared error: MSE
            - Categorical cross entropy: CCE
    - the commit of the repo with the source code

Seeing the tensorboard (example)
- BASH
	- mkdir logs
	- example: logs/20180716_041926/events.out.tfevents.1531707586.runner-16.dei.unipd.it
	- mkdir 20180716_041926
	- scp prosperolo@login.dei.unipd.it:/nfsd/hda/prosperolo/HDACostaProspero/logs/20180716_041926/* ./20180716_041926
- CONDA
	- tensorboard --logdir 20180716_041926
	- http://PC-NAME:6006

Useful commands
- Create a new repository
	- git clone https://gitlab.dei.unipd.it/costarob/HDACostaProspero.git
	- cd HDACostaProspero
- Avoid the need of typing password
	- git remote set-url origin https://USERNAME:PASSWORD@gitlab.dei.unipd.it/costarob/HDACostaProspero.git
- Go back to a previous commit
    - git log
        - find the HEAD commit and the DESIRED commit to go back to
    - git revert --no-commit DESIRED..HEAD
    - git commit -m "reset"
    - git push
    
- Remove files
	- git rm file1.txt
	- git commit -m "remove file1.txt"
	- git push

- Undo git add
	- git reset file1.txt
- Existing folder
	- cd existing_folder
	- git init
	- git remote add origin https://gitlab.dei.unipd.it/costarob/HDACostaProspero.git
	- git add .
	- git commit -m "Initial commit"
	- git push -u origin master

- Existing Git repository
	- cd existing_repo
	- git remote rename origin old-origin
	- git remote add origin https://gitlab.dei.unipd.it/costarob/HDACostaProspero.git
	- git push -u origin --all
	- git push -u origin --tags
