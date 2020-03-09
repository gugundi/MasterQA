# MasterQA
Master Thesis project on Question Answering

## Connecting to DTU server
1. Download [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html "PuTTY download link")
2. Open PuTTY and type 'thinlinc.compute.dtu.dk' in hostname.
3. Open connection and type in DTU credentials
4. Once connected open an ssh tunnel to one of the [DTU GPU clusters](https://itswiki.compute.dtu.dk/index.php/GPU_Cluster "DTU cluster machine - hostname"). Names can be found under Cluster Machines - Hostname. Example: 'ssh themis'.
5. Enter DTU password again. You are now connected to the DTU cluster.

## First time setup (Break)
1. First we need to setup a new conda environment for development. This can be done using the command 'conda create --name $name_of_choosing python=3.6.8'. The Break code is developed on python version 3.6.8. 
2. Activate environment with 'conda activate $name_of_choosing'
3. Clone [Break code](https://github.com/tomerwolgithub/Break "Break code") and download the [Break dataset](https://github.com/allenai/Break/raw/master/break_dataset/Break-dataset.zip "Break dataset direct download"). It might be a good idea to create a seperat project folder for this.
4. Navigate to ./Break/qdmr_parsing/ and install the required packages with 'pip install -r requirements.txt'. This might take a couple of minutes. 
5. Next download spacy 'python -m spacy download en_core_web_sm'.

## Running QDMR Parsing
Follow instructions on [QMDR Parsing Github page](https://github.com/tomerwolgithub/Break/tree/master/qdmr_parsing "Github link"). 
