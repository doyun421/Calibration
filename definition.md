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

                      Tangential distortion occurs because the image-taking lens is not aligned perfectly parallel to the imaging plane. 
                      x_distorted = x + [2p_1*xy + p_2(r^2 + 2x^2)]
                      y_distorted = y + [p_1(r^2 + 2y^2) + 2p_2*xy]

                      Distortion coefficients = (k1, k2, p1, p2, k3)
                      
                                    * Pinhole camera: 
                                                      is a simple camera without a lens but with a tiny aperture (the so-called pinhole) effectively a light-proof box with a small hole in one side.
                                    * Kn is the n_th radial distortion coefficient
                                    * Pn is the n_th tangential distortion coefficient


                      Focal length (fx, fy) and optical centers (cx, cy) can be used to create a camera matrix(intrinsic parameters)
                      Extrinsic parameters corresponds to rotation and translation vectors which translates a coordinates of a 3D point to a coordinate system.


                      Pick a reference point in the 3D world, mark it as the origin and define the world coordinate system axis. Rotate and translate the wolrd coordinate system to the camera coordinate system. A 3D point defined in the world coordinate system will now be in the camera coordinate system. 

                      Notice that by [x, y, z] facilitates (possible 가능하게 하다) the dot product to obtain the camera coordinates of the point in 3D space. R refers to the rotation matrix and t refers to a translation matrix that first rotates a point to camera coordinate system orientation and translate it to camera coordinate system. [R|T] is also called the extrinsic camera matrix that rotates and translate object in the specified world coordinate system to the camera coordinate system. 



                    ### Homogeneous Coordinates
                    [x, y, z] and [Xc, Yc, Zc] are called homogeneous coordinates and such camera matrix transformation is a projective transformation described by homogeneous coordinates. A three dimensional world point is represented by four homogeneous coordinates with the last coordinate. A two dimensional point on the image is represented by three homogenous coordinates with the last coordinate the depth(Z axis) of the point. To better understand homogenous coordinates, we will use the two dimensional image point represented by three homogenous coordinates as an example since three dimension can be depicted in drawings. A point on the image (0, 200) can be seen as a ray from the origin as (0, 200, 1).






                ## Intrinsic Matrix
                      shift origin of image plane to top left. 
                      A point defined in the camera coordinate system can be projected into the image plane. 
                      that involves fx, fy that scales the x and y values of camera coordinate system to u and v values of the image plane. which translate the origin of the image from the centre to the top left corner of the image. The full camera matrix, P, that takes in the world coordinate point and project it to the image plane with the full formula. Since K is a 3x3 matrix and [R|t] is a 3x4 matrix, P is a 3x4 matrix. As P is not a square matrix, its matrix inverse is not possible and hence this again shows the difficulty in back calculating the x,y,z world coordinates with u,v pixels of camera image. Even if P inverse is possible, an image coordinate only contains u and v, the absence of Zc, depth of image, would make inverse calculation (P inverse)(Zc)([u v 1]) impossible.


                    ### Homography
                    However, there is a technique called homography to recover 3D position from image pixels when the z direction in the world coordinate system is ignored. In another words, we consider only a plane in the 3D world. If the z direction in the world coordinate is ignored, the 4x3 camera matrix, P, can be reduced to a 3x3 homography matrix, H. A square matrix is able to have its inverse, H-1, which can map a u, v pixel of a image to a x,y, 0 coordinate in the world coordinate system as shown below.
                    In fact, image to image mapping is also possible as you can define the first image in the world coordinates with z=0. Such a technique is often used in telecast of swimming competition when a image of national flags is transformed onto swimming pool lanes.


                ### Inverse Mapping
                Distance is distorted in a perspective view as a fixed distance nearer to the camera appear bigger and same distance further away from the camera appear smaller.  to rectify the image from a perspective view to a top-down orthogonal view to measure distance (https://arxiv.org/pdf/1905.02231.pdf).

                Given an image that is captured by a camera that is pitch at at angle, first get the camera coordinates, then rotate the camera axis around the camera coordinates x axis to face perpendicular to the ground, and re-project the rotated camera coordinates onto the image plane.




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
                  

          














                  
                  
just do only what can you do and then do all things left.
for fast process, you must skip what you didn't know. 
later you can find what was broken had been solved. 

                  1. find all you can do. and do!
                  2. retry what you couldn't do last time. - find all thing you can do.
                  3. extremely you couldn't know info --> you can ask professor or other expert. 

                  4. accumulate all info from papers, codes, gits, or competitons, else, books, youtube, else. in a week. 
                  9.21 - 9.28 
                  and apply most famous school or company or business for earning money..
                  
                  
                      
      

# 2. Understanding the Construction of Vanishing Point and Horizon
# 3. Recovering Extrinsic Rotation Matrix with Vanshing Points

