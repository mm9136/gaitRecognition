# gaitRecognition

### Project Content

Project is done in PyCharm and contains following script (that do not depend one on each other):

- GaitRecognition - main scipt, just by running it detects gait using MediaPipe, create json files and it use ORB detector as well

- Basic.py - detects gait using MediaPipe

- Sift.py - implementation of sift detector 

- Orb.py - implementation of orb detector

- SiftLiveCamera - implementation of sift detector in live captured video


### Installation

Installing and setting up 

1. Install PyCharm

2. Clone the repo and open project in PyCharm
   ```sh
   git clone https://github.com/mm9136/gaitRecognition.git
   ```
3. Before running the project it is necessary to install the specified packages:
   * OpenCV
  ```sh
  pip install opencv-python
  ```
  
  * MediaPipe
  ```sh
  pip install mediapipe
  ```
  
  * Keyboard
  ```sh
  pip install keyboard
  ```
4. Run GaitRecognition.py script 
   





