# This script is for downloading the GQA dataset
mkdir -p dataset
cd dataset

# Get Scene Graphs
mkdir scene_graphs
cd scene_graphs
wget https://nlp.stanford.edu/data/gqa/sceneGraphs.zip
unzip sceneGraphs.zip
cd ..

# Get Questions
mkdir questions
cd questions
wget https://nlp.stanford.edu/data/gqa/questions1.3.zip
unzip questions1.3.zip
cd ..

# Get Images
mkdir images
cd images
wget https://nlp.stanford.edu/data/gqa/allImages.zip
unzip allImages.zip
