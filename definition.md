# Purpose of Robotics Camera Calibration.

## Calibration
  1. Detecting correct path's line with 98% accuracy. 
      How to improve accuracy of detection.
      

      Accuracy Standarzation
         = Number of correct predictions / Total number of predictions
         = TP + TN / (TP + TN + FP + FN) 
         code : tn, fn, tp, fp = confusion_matrix(y_test, y_pred).ravel() 
                print ((tp+tn) / (tn+fn+tp+fp)) 
         
     
  3. Preventing accidents related with calibration inaccuracy causing issues. 
  4. High rates of rejected parts (Loss of finance) 
  5. Cost and time overruns 
  6. Avoiding customers inefficient additional requirements. 
  7. Preventing reputation damage.


### Types of Issues of Calibration

      a) Weather changes.
      b) Network Issues. 
      c) unpredicted visual affection. 
      d) embedded system's issue causing. 

Milestone



## Lane Detection


## Traffic Sign Detection


### Calibration for autonomous driving in road scenarios
      
      1) Due to the vibration (sensors it hard to realize in users' hands) and operating condition changes in daily use(weather changes, what factors would be considered to change calibration-e.g. temperature, humidity, pressure, vibration, noise, dust, electromagnetic interference), extrinsic parameter of sensors won't stay the same. 
      2) SensorX2car, containing online calibration methods for four commonly-used sensors: camera, LiDAR, GNSS/INS(Inertial Navigation System) device, and 2D millimeter wave radar. 
      3) For the camera, they use a deep learning network to predict the vanishing point ![image](https://github.com/doyun421/Calibration/assets/73266189/48ccec76-b824-4281-a437-5a3d0b56c1f1)
(a point at which something disappears or ceases to exist) and horizon line of a single image and then convert them to the roll, pitch, and yaw angles ![image](https://github.com/doyun421/Calibration/assets/73266189/c23ccda1-9aae-4bb7-8db1-f6ae9baa3dbe)
      4) 

# 1. A Single Camera 3D Functions
      1) We have to understand the basics of how a camera turn a 3D scene into a 2D image. 
      camera coordinate system where the camera sensor position is the origin (0, 0, 0) and the x, y, z axis of the camera coordinate system. 

      v/fy = y/z
      x/fx = x/z

![calibration_camera_coordinate](https://github.com/doyun421/Calibration/assets/73266189/80377570-67c5-41bb-97f7-d263c678294b)

      2) Extrinsic Matrix
              Goal: types of distortion caused by cameras (Radial distortion-barrel distortion, Pincushion distortion, Mustache distortion.), (tangential distortion(De-centering), compound distortions which combines barrel, pincushion, and tangential effects)
                    how to find the intrinsic and extrinsic properties of a camera
                    how to undistort images based off these properties
              Basics: 
      

# 2. Understanding the Construction of Vanishing Point and Horizon
# 3. Recovering Extrinsic Rotation Matrix with Vanshing Points

