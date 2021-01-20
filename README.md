# object_tracking
This repo contains C++ examples to use the following object trackers:
- Multi-Scale KCF 
- Multi-Scale MOSSE (from [Dlib](https://github.com/davisking/dlib))

Also, there is another tracker that I implemented for ARM CPUs. It extracts FAST keypoints from the detected object, tracks them by KLT algorithm, and maps the bounding box to a new box using the estimated similarity transformation between the points. 

In addition to being multi-scale, all the trackers would report failures when the object is no longer in the scene or the tracking quality is below a threshold. This makes them very applicable to real-world projects. Feel free to use and send me PRs if you found better trackers :)

# How to Build?
Go inside each folder and then:
```
mkdir build
cd build
cmake ..
make
```
Run the generated executable and enjoy! It detects face in the video using a pre-trained SSD model and then tries to track it in the subsequent frames. 

![](https://j.gifs.com/mO1JX3.gif)

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
