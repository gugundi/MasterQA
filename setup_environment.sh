#update conda
#conda update -n base -c defaults conda
#conda update conda

#create envirorment named notebook
conda create --name notebook python pip

#conda init zsh
#activate envirorment
conda activate notebook

#install requirements
#while read requirement; do conda install --yes $requirement; done < requirements.txt
pip install -r requirements.txt

#install jupyter
conda install -c conda-forge jupyter-lab

#install envirorment
#conda install ipykernel
#python -m ipykernel install --user --name notebook --display-name "Python (testEnv)"
