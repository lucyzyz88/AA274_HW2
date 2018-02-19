#!/usr/bin/python

############################################################
# ExtractLines.py
#
# This script reads in range data from a csv file, and
# implements a split-and-merge to extract meaningful lines
# in the environment.
############################################################

# Imports
import numpy as np
from PlotFunctions import *


############################################################
# functions
############################################################

#-----------------------------------------------------------
# ExtractLines
#
# This function implements a split-and-merge line
# extraction algorithm
#
# INPUT: RangeData - (x_r, y_r, theta, rho)
#                x_r - robot's x position (m)
#                y_r - robot's y position (m)
#              theta - (1D) np array of angle 'theta' from data (rads)
#                rho - (1D) np array of distance 'rho' from data (m)
#           params - dictionary of parameters for line extraction
#
# OUTPUT: (alpha, r, segend, pointIdx)
#         alpha - (1D) np array of 'alpha' for each fitted line (rads)
#             r - (1D) np array of 'r' for each fitted line (m)
#        segend - np array (N_lines, 4) of line segment endpoints.
#                 each row represents [x1, y1, x2, y2]
#      pointIdx - (N_lines,2) segment's first and last point index

def ExtractLines(RangeData, params):

    # Extract useful variables from RangeData
    x_r = RangeData[0]
    y_r = RangeData[1]
    theta = RangeData[2]
    rho = RangeData[3]


    ### Split Lines ###
    N_pts = len(rho)
    r = np.zeros(0)
    alpha = np.zeros(0)
    pointIdx = np.zeros((0, 2), dtype=np.int)

    # This implementation pre-prepartitions the data according to the "MAX_P2P_DIST"
    # parameter. It forces line segmentation at sufficiently large range jumps.
    rho_diff = np.abs(rho[1:] - rho[:(len(rho)-1)])
    LineBreak = np.hstack((np.where(rho_diff > params['MAX_P2P_DIST'])[0]+1, N_pts))
    startIdx = 0
    for endIdx in LineBreak:
        alpha_seg, r_seg, pointIdx_seg = SplitLinesRecursive(theta, rho, startIdx, endIdx, params)
        #print(endIdx)
        N_lines = r_seg.size
        print(N_lines)

        ### Merge Lines ###
        if (N_lines > 1):
            alpha_seg, r_seg, pointIdx_seg = MergeColinearNeigbors(theta, rho, alpha_seg, r_seg, pointIdx_seg, params)
        r = np.append(r, r_seg)
        alpha = np.append(alpha, alpha_seg)
        pointIdx = np.vstack((pointIdx, pointIdx_seg))
        startIdx = endIdx

    N_lines = alpha.size

    ### Compute endpoints/lengths of the segments ###
    segend = np.zeros((N_lines, 4))
    seglen = np.zeros(N_lines)
    for i in range(N_lines):
        rho1 = r[i]/np.cos(theta[pointIdx[i, 0]]-alpha[i])
        rho2 = r[i]/np.cos(theta[pointIdx[i, 1]-1]-alpha[i])
        x1 = rho1*np.cos(theta[pointIdx[i, 0]])
        y1 = rho1*np.sin(theta[pointIdx[i, 0]])
        x2 = rho2*np.cos(theta[pointIdx[i, 1]-1])
        y2 = rho2*np.sin(theta[pointIdx[i, 1]-1])
        segend[i, :] = np.hstack((x1, y1, x2, y2))
        seglen[i] = np.linalg.norm(segend[i, 0:2] - segend[i, 2:4])

    ### Filter Lines ###
    # Find and remove line segments that are too short
    goodSegIdx = np.where((seglen >= params['MIN_SEG_LENGTH']) &
                          (pointIdx[:, 1] - pointIdx[:, 0] >= params['MIN_POINTS_PER_SEGMENT']))[0]
    pointIdx = pointIdx[goodSegIdx, :]
    alpha = alpha[goodSegIdx]
    r = r[goodSegIdx]
    segend = segend[goodSegIdx, :]

    # change back to scene coordinates
    segend[:, (0, 2)] = segend[:, (0, 2)] + x_r
    segend[:, (1, 3)] = segend[:, (1, 3)] + y_r

    return alpha, r, segend, pointIdx


#-----------------------------------------------------------
# SplitLineRecursive
#
# This function executes a recursive line-slitting algorithm,
# which recursively sub-divides line segments until no further
# splitting is required.
#
# INPUT:  theta - (1D) np array of angle 'theta' from data (rads)
#           rho - (1D) np array of distance 'rho' from data (m)
#      startIdx - starting index of segment to be split
#        endIdx - ending index of segment to be split
#        params - dictionary of parameters
#
# OUTPUT: alpha - (1D) np array of 'alpha' for each fitted line (rads)
#             r - (1D) np array of 'r' for each fitted line (m)
#           idx - (N_lines,2) segment's first and last point index

def SplitLinesRecursive(theta, rho, startIdx, endIdx, params):

    ##### TO DO #####
    # Implement a recursive line splitting function
    # It should call 'FitLine()' to fit individual line segments
    # In should call 'FindSplit()' to find an index to split at
    #################
    #Current Data Sets to fit
    theta_Current = theta[startIdx:endIdx]
    rho_Current = rho[startIdx:endIdx]
    #Current Idx
    Idx_current = startIdx, endIdx
    alpha = np.zeros(1)
    r = np.zeros(1)
    idx = np.zeros((0,2),dtype=int)
    

    #Fit the line for the data points
    alpha_current, r_current = FitLine(theta_Current, rho_Current)
    #Find the split point in this data points set
    P = FindSplit(theta_Current, rho_Current, alpha_current, r_current, params)
    #If it's not mergable
    if (P == -1 | endIdx - startIdx < 1):
        print 'reach bottom'
        alpha = alpha_current
        r = r_current
        idx = Idx_current
        return alpha, r , idx
    else:
        #Split L into L1 & L2 for two subsets, and do recursion on those
        startIdx_left = startIdx
        endIdx_left = startIdx + P
        startIdx_right = startIdx + P + 1
        endIdx_right = endIdx
        print 'P in splitrecursive', P 
        alpha_left, r_left, idx_left = SplitLinesRecursive(theta,rho,startIdx_left,endIdx_left,params)
        alpha_right, r_right, idx_right = SplitLinesRecursive(theta,rho,startIdx_right,endIdx_right,params)
        np.append(alpha,alpha_left)
        np.append(alpha,alpha_right)
        np.append(r,r_left)
        np.append(r,r_right)
        np.vstack((idx,idx_left))
        np.vstack((idx,idx_right))
    return alpha, r, idx


#-----------------------------------------------------------
# FindSplit
#
# This function takes in a line segment and outputs the best
# index at which to split the segment
#
# INPUT:  theta - (1D) np array of angle 'theta' from data (rads)
#           rho - (1D) np array of distance 'rho' from data (m)
#         alpha - 'alpha' of input line segment (1 number)
#             r - 'r' of input line segment (1 number)
#        params - dictionary of parameters
#
# OUTPUT: SplitIdx - idx at which to split line (return -1 if
#                    it cannot be split)

def FindSplit(theta, rho, alpha, r, params):

    ##### TO DO #####
    # Implement a function to find the split index (if one exists)
    # It should compute the distance of each point to the line.
    # The index to split at is the one with the maximum distance
    # value that exceeds 'LINE_POINT_DIST_THRESHOLD', and also does
    # not divide into segments smaller than 'MIN_POINTS_PER_SEGMENT'
    # return -1 if no split is possiple
    #################
    n = np.shape(theta)[0]; #n is the number of points in the subset
    print 'n!!!' , n
    Distance = np.absolute(rho*np.cos(theta - alpha) - r)
    #Find the best split but excluding the start and end for min_point_segment
    min_point_segment = params['MIN_POINTS_PER_SEGMENT']
    Max_Split_Dist = 0
    Max_DIST = 0
    splitIdx = 0

    if (n > 2*params['MIN_POINTS_PER_SEGMENT']):
        # Max_Split_Dist = np.max(Distance[min_point_segment:(n-min_point_segment)])
        splitIdx = np.argmax(Distance)
        Max_DIST = np.max(Distance)
        if (splitIdx < min_point_segment):
            splitIdx = min_point_segment + np.argmax(Distance[min_point_segment:(n-min_point_segment)])
            Max_DIST = np.max(Distance[min_point_segment:(n-min_point_segment)])
    else:
        splitIdx = -1

    if (Max_DIST < params['LINE_POINT_DIST_THRESHOLD']):
        splitIdx = -1
    if (splitIdx < params['MIN_POINTS_PER_SEGMENT'] | splitIdx > (n-params['MIN_POINTS_PER_SEGMENT'])):
        splitIdx = -1
    print 'splitIDX in split', splitIdx
    # If the size of the segment is too small
    # elif (((splitIdx + 1) < params['MIN_POINTS_PER_SEGMENT'])| (n - splitIdx) < params['MIN_POINTS_PER_SEGMENT']): #split at the spliIdx
    #     splitIdx = -1

    return splitIdx


#-----------------------------------------------------------
# FitLine
#
# This function outputs a best fit line to a segment of range
# data, expressed in polar form (alpha, r)
#
# INPUT:  theta - (1D) np array of angle 'theta' from data (rads)
#           rho - (1D) np array of distance 'rho' from data (m)
#
# OUTPUT: alpha - 'alpha' of best fit for range data (1 number) (rads)
#             r - 'r' of best fit for range data (1 number) (m)

def FitLine(theta, rho):

    #### TO DO #####
    #Implement a function to fit a line to polar data points
    #based on the solution to the least squares problem (see Hw)
    ################
    alpha1 = 0
    alpha2 = 0
    alpha3 = 0
    alpha4 = 0
    n = theta.size
    r_sum = 0
    print 'n in fitline', n
    for i in range(n):
        alpha1 = alpha1 + (rho[i]**2)*np.sin(2*theta[i])
        alpha3 = alpha3 + (rho[i]**2)*np.cos(2*theta[i])
        for j in range(n):
            alpha2 = alpha2 + (rho[i] * rho[j] * np.cos(theta[i]) * np.sin(theta[j]))
            alpha4 = alpha4 + (rho[i] * rho[j] * np.cos(theta[i]+theta[j]))

    alpha = 0.5 * np.arctan2((alpha1- 2*alpha2/n),(alpha3 - alpha4/n)) + 1.57  
    
    for i in range(n):
        r_sum = r_sum + rho[i]*np.cos(theta[i] - alpha)
    r = r_sum/n
    r = 1./theta.size * np.sum(rho * np.cos(theta - alpha))
    return alpha, r


#---------------------------------------------------------------------
# MergeColinearNeigbors
#
# This function merges neighboring segments that are colinear and outputs
# a new set of line segments
#
# INPUT:  theta - (1D) np array of angle 'theta' from data (rads)
#           rho - (1D) np array of distance 'rho' from data (m)
#         alpha - (1D) np array of 'alpha' for each fitted line (rads)
#             r - (1D) np array of 'r' for each fitted line (m)
#      pointIdx - (N_lines,2) segment's first and last point indices
#        params - dictionary of parameters
#
# OUTPUT: alphaOut - output 'alpha' of merged lines (rads)
#             rOut - output 'r' of merged lines (m)
#      pointIdxOut - output start and end indices of merged line segments

def MergeColinearNeigbors(theta, rho, alpha, r, pointIdx, params):

    ##### TO DO #####
    # Implement a function to merge colinear neighboring line segments
    # HINT: loop through line segments and try to fit a line to data
    #       points from two adjacent segments. If this line cannot be
    #       split, then accept the merge. If it can be split, do not merge.
    #################
    #This gives how many line segmments are there
    N_lines, _ = np.shape(pointIdx) 
    pointIdxOut = []
    alphaOut = []
    rOut = []
    for i in range(N_lines - 2):
        [StartIdx_left, EndIdx_left] = pointIdx[i]
        [StartIdx_right, EndIdx_right] = pointIdx[i+1]
        pointIdx_current = [StartIdx_left,EndIdx_right]
        alpha_fit, r_fit = Fitline(theta[StartIdx_left:EndIdx_right].ravel,rho[StartIdx_left:EndIdx_right].ravel)
        Split_Result = FindSplit(theta[StartIdx_left:EndIdx_right].ravel, rho[StartIdx_left:EndIdx_right].ravel, alpha_fit, r_fit, params)
        if(Split_Result == -1):
            np.append(alphaOut, alpha_fit)
            np.append(rOut, r_fit)
            np.concatenate(pointIdx,pointIdx_current)
            #Because it merges with the next one, therefore skip the next i
            i = i+2
        else:
            np.append(alphaOut, alpha[i])
            np.append(rOut, r[i])
            np.concatenate(pointIdxOut, pointIdx[i])
            #If this was the second last point and it didn't merge with 
            #the last segment, since it won't iterate again, 
            #therefore add the last point here
            if(i == Nlines - 2):
                alphaOut[i+1] = alpha[i+1]
                rOut[i+1] = alpha[i+1]
                pointIdxOut[i+1] = pointIdx[i+1]
            i = i+1
    return alphaOut, rOut, pointIdxOut
#----------------------------------
# ImportRangeData
def ImportRangeData(filename):

    data = np.genfromtxt('./RangeData/'+filename, delimiter=',')
    x_r = data[0, 0]
    y_r = data[0, 1]
    theta = data[1:, 0]
    rho = data[1:, 1]
    return (x_r, y_r, theta, rho)
#----------------------------------


############################################################
# Main
############################################################
def main():
    # parameters for line extraction (mess with these!)
    MIN_SEG_LENGTH = 0.05  # minimum length of each line segment (m)
    LINE_POINT_DIST_THRESHOLD = 0.1  # max distance of pt from line to split
    MIN_POINTS_PER_SEGMENT = 2  # minimum number of points per line segment
    MAX_P2P_DIST = 0.35  # max distance between two adjent pts within a segment

    # Data files are formated as 'rangeData_<x_r>_<y_r>_N_pts.csv
    # where x_r is the robot's x position
    #       y_r is the robot's y position
    #       N_pts is the number of beams (e.g. 180 -> beams are 2deg apart)

    #filename = 'rangeData_5_5_180.csv'
    #filename = 'rangeData_4_9_360.csv'
    filename = 'rangeData_7_2_90.csv'

    # Import Range Data
    RangeData = ImportRangeData(filename)

    params = {'MIN_SEG_LENGTH': MIN_SEG_LENGTH,
              'LINE_POINT_DIST_THRESHOLD': LINE_POINT_DIST_THRESHOLD,
              'MIN_POINTS_PER_SEGMENT': MIN_POINTS_PER_SEGMENT,
              'MAX_P2P_DIST': MAX_P2P_DIST}

    alpha, r, segend, pointIdx = ExtractLines(RangeData, params)

    ax = PlotScene()
    ax = PlotData(RangeData, ax)
    ax = PlotRays(RangeData, ax)
    ax = PlotLines(segend, ax)

    plt.show(ax)

############################################################

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
