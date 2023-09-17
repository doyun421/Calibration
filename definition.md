# Purpose of Robotics Camera Calibration.

## Calibration
  1. Detecting correct path's line with 98% accuracy. 
      How to improve accuracy of detection.
      

      Accuracy Standarzation
         = Number of correct predictions / Total number of predictions
         = TP + TN / (TP + TN + FP + FN) 
         code : tn, fn, tp, fp = confusion_matrix(y_tests, y_pred).ravel() 
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
                      Radial distortion causes straight lines to appear curved. 
                      Radial distortion occurs because the lens has bad quality.
                      x_distorted = x(1 + k_1*r^2 + k_2*r^4 + k_3*r^6)
                      y_distorted = y(1 + k_1*r^2 + k_2*r^4 + k_3*r^6)

                      Tangential distortion occurs because the image-taking lense is not aligned perfectly parallel to the imaging plane. 
                      x_distorted = x + [2p_1*xy + p_2(r^2 + 2x^2)]
                      y_distorted = y + [p_1(r^2 + 2y^2) + 2p_2*xy]

                      Distortion coefficients = (k1, k2, p1, p2, k3)
                      
                                    * Pinhole camera: 
                                                      is a simple camera without a lens but with a tiny aperture (the so-called pinhole) eddectively a light-proof box with a small hole in one side.
                                    * Kn is the n_th radial distortion coefficient
                                    * Pn is the n_th tangential distortion coefficient


                      Focal length (fx, fy) and optical centers (cx, cy) can be used to create a camera matrix(intrinsic parameters)
                      Extrinsic parameters corresponds to rotation and translation vectors which translates a coordinates of a 3D point to a coordinate system.

                      

### code
        Setup
              3D points are called object points and 2D image points are called image points.
              Finds the positions of internal corners of the chessboard. 
              
        1. termination criteria (기준) 
                               The class defining termination criteria for iterative algorithms. 
                               enum is a special class that represents a group of constants.
                               enum type {
                                          COUNT = 1, 
                                          MAX_ITER = COUNT, 
                                          EPS = 2 (epsilon)
                                          }
                              Stop the algorithm iteration if specified accuracy, epsilon, is reached.
                              Stop the iteration when any of the aboe condition is met
                              max_iter: An integer specifying maximum number of iterations.
                              epsilon: Required accuracy.
                              
                              cv::TermCriteria::TermCriteria ( int type, 
                                                               int maxCount,
                                                               double epsilon
                                                               )
                              cv::TermCriteriaMaxIter
                              https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html

          2. prepare object points, like (0,0,0), (1,0,0), (2,0,0), ... , (6,5,0)
                              np.zeros
                                      shape: int or tuple of ints
                                      e.g., (2, 3) or (2)
                                      dtype : data-type, optional 
                                      numpy.int. Default is numpy.float64.
                                      order: Whether to store multi-dimensional data in row-ajor (C-style) or column-major (Fortran-style) order in memory.
                                      e.g., np.zeros((2, 4), dtype=int)
                                      array([[ 0., 0. ], 
                                             [ 0., 0. ], 
                                             [ 0., 0. ],
                                             [ 0., 0. ]])

                              np.meshgrid ( != mgrid)
                                      returns a grid that is useful in plotting contour plots of 3D graphs.
                                      1. The input array to make a grid out of
                                      2. If you want a 2D grid
                                      3. Whether to return a sparse grid. Set this to True if you'r dealing with large volumes of data that cannot be fit into memeory.
                                      4. 
                                      
                                      x1, x2, ... , x: array_like
                                            1-D arrays representing the coordinates of grid.
                                      indexing: {'xy', 'ij'}
                                            Cartesian ('xy', default) or matrix('ij')indexing of output.

                              np.mgrid
                                      An instance which returns a dense (or fleshed out) mesh-grid when indexed, 
                                      index an indicator, sign, or measure of something.
                                      link the value of automatically to the value of a price index.
                                      If the step length is a complex number (e.g. 5j), then the integer part of its magnitude is interpreted asaspecifying the number of points to create between the starat and stop values, where the stop value is inclusive. 
                                      np.mgrid[0:5, 0:5] 격자, 눈금
                                      = array([[[0, 0, 0, 0, 0],
                                                [1, 1, 1, 1, 1],
                                                [2, 2, 2, 2, 2],
                                                [3, 3, 3, 3, 3],
                                                [4, 4, 4, 4, 4]],
                                               [[0, 1, 2, 3, 4], 
                                                [0, 1, 2, 3, 4],
                                                [0, 1, 2, 3, 4],
                                                [0, 1, 2, 3, 4],
                                                [0, 1, 2, 3, 4]]])
                                      np.mgrid[-1:1:5j]
                                      = array([-1., -0.5, 0., 0.5, 1. ])





                                      
                              numpy.linspace
                                      Return evenly spaced numbers over a specified interval.
                                      np.linspace(2.0, 3.0, num=3)
                                      = array([2., 2.5, 3. ])
                                      --
                                            
                  
                              
               3. Arrays to store object and image points from all the images. 
               4. Find the chess board corners
                             Finds the positions of internal corners of the chessboard 
                             A regular chessboard has 8*8 sqauares and 7*7 internal corners
                             Renders the detected chessboard corners. 
                             * patternSize: Number of inner corners per a chessboard row and column
                             ![image](https://github.com/doyun421/Calibration/assets/73266189/b7f5066b-7e8e-4969-b05a-6a5c71295038)

               5. If found, add object points, image points
 
        ### Undistortion
            Refine the camera matrix based on a free scaling parameter (Scale parameter: probability distibutions is such that there is a parameter s (and other parameter theta) for which the cumulative distribution function satisfies 
                              

            
        1. Using cv.undistort()
                Just call the function and use ROI (Region of Intereset, ROI) obtained above to crop the result. 
                * ROI 
                  일몰 사진 안에서 관심 영역인 태양 주위를 초록색 사각형으로 표시한 이미지. 

                  "s" is called a scale parameter, since its value determines the "scale" or statistical disperison of the probability distribution. If s is large, then the distribution will be more spread out; if s is small then it will be more concentrated. 




                  # cumulative distribution function (CDF) of a real-valued random variable X, or just distribution function of X, evaluated at x, is the probability that X will take a value less than or equal to x. Every probability distribution supported on the real numbers, 
                  
                  *(dispersion (also called variability, scatter, or spread) is the extent to which a distribution is stretched or squeezed (stretch shows opposite tendencies related with squeezed). 
                  
                        Example of samples from two populations with the same mean but different dispersion. The blue population is much more dispersed than the red population. 
                  * Support
                  In mathematics, the support (sometimes topological support or spectrum) of a measure u on a measurable topological space.  
                  
                  
                  f_s(x) = F(x/s)/s,
                  f(x) same f_(s=1)(x).



                  I tried hardly search what I need but I will try other way as create own creatures. 




                  


                  
                      
      

# 2. Understanding the Construction of Vanishing Point and Horizon
# 3. Recovering Extrinsic Rotation Matrix with Vanshing Points

