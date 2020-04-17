# rawLab
Playing with raw image files.  
rawLab consists in a main scipt rawLab.py  and some functions files (imgIO, imgModification, imgFilter and imgConstant)  
Feel free to experiment and modify them as you want !


# Requirements
Python distribution : Miniconda (https://docs.conda.io/en/latest/miniconda.html) or Anaconda  
6Gb of RAM (8+ is better)  
GPU supporting CUDA with 2Gb of VRAM (optional)  

# How to install required Python packages 
Open conda command prompt and enter the following (line by line) :  
pip install rawpy  
conda install numpy scipy pillow matplotlib  
conda install pytorch cudatoolkit=10.1 -c pytorch  
conda install spyder (optional IDE)  

# How to use  
In command prompt : python rawLab.py  
In Spyder : just open and run rawLab.py  
It will open sample.CR2 (grayscale), convert it to bayer RGB and demosaicing it using linear interpolation. One image is displayed at each step.  

# Notes  
White balancing is not finished at the end of the script.  
  
