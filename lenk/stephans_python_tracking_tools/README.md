# Python routines for cloud tracking

This git contains python routines needed to track clouds. Currently four tracking types are supported:

1. object tracking with hysteresis thresholding, watershed segmentation or a segmentation using morphological reconstruction
2. cross correlation tracking
3. optical flow tracking after Farnebäck (2003)
4. Tv-L1 optical flow trcaking after Zach et al. (2007)

Currently all the routines are designed for the use with a fixed track structure. From a given starting point or object, tracks are
created for half an hour backward and forward in time, only.
