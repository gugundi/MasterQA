# This script is for downloading the GQA dataset
mkdir -p dataset
cd data

# Get Scene Graphs
wget https://nlp.stanford.edu/data/gqa/sceneGraphs.zip
unzip sceneGraphs.zip

# Get Questions
wget https://nlp.stanford.edu/data/gqa/questions1.3.zip
unzip questions1.3.zip
