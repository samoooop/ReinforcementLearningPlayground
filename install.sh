sudo apt-get install python3-pip git zlib1g-dev cmake swig
pip3 install virtualenv
cd ~
mkdir .pythonenv
cd .pythonenv
virtualenv .
echo 'source ~/.pythonenv/bin/activate' >> ~/.bashrc
pip3 install 'gym[all]' opencv-pyhton tensorflow-gpu
cd ~/Documents
git clone https://github.com/openai/baselines
cd baselines
pip3 install -e .

