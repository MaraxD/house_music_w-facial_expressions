### Have you ever thought of creating house music using your facial expressions? Me neither.
# Setup
In order for this script to detect facial expressions, i used the Haar-cascade detection algorithm. <br>
I have used an already trained model (model.h5) with a training accuracy of 72% and an accuracy of 60%. Next up, it was time to use detection on our model, alongside openCV. This allows me to capture and detect facial expression in real time.<br>
The detected faces are then put through the emotion detection model and based on the detected emotion, a house sound snippet will play. 

# Try it yourself
Start by cloning this project and installing dependencies using 
``pip install -r requirements.txt
``
Simply run the script and enjoy your own music !

<br><br>
Code inspiration taken from @SHAIK-AFSANA


