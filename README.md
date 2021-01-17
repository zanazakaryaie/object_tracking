# object_tracking
Three object trackers you deserve!

Go inside each folder and then:
```
mkdir build
cd build
cmake ..
make
```
Run the generated executable and enjoy!

# Note
If you don't have Dlib on your computer, install it via:
```
sudo apt-get install build-essential cmake pkg-config 
sudo apt-get install libx11-dev libatlas-base-dev 
sudo apt-get install libgtk-3-dev libboost-python-dev 

git clone https://github.com/davisking/dlib.git 
cd dlib 
mkdir build 
cd build 
cmake .. cmake --build . --config Release 
sudo make install 
sudo ldconfig
```

# Appendix
For more details read my post [here](http://imrid.net/?p=4441).
