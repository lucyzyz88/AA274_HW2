#!/usr/bin/python

import time
import os

import cv2
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
import numpy as np
import glob

import pdb

from camera_calibration.calibrator import MonoCalibrator, ChessboardInfo, Patterns


class CameraCalibrator:

    def __init__(self):
        self.calib_flags = 0
        self.pattern = Patterns.Chessboard

    def loadImages(self, cal_img_path, name, n_corners, square_length, n_disp_img=1e5, display_flag=True):
        self.name = name
        self.cal_img_path = cal_img_path

        self.boards = []
        self.boards.append(ChessboardInfo(n_corners[0], n_corners[1], float(square_length)))
        self.c = MonoCalibrator(self.boards, self.calib_flags, self.pattern)

        if display_flag:
            fig = plt.figure('Corner Extraction', figsize=(12, 5))
            gs = gridspec.GridSpec(1, 2)
            gs.update(wspace=0.025, hspace=0.05)

        for i, file in enumerate(sorted(os.listdir(self.cal_img_path))):
            img = cv2.imread(self.cal_img_path + '/' + file, 0)     # Load the image
            img_msg = self.c.br.cv2_to_imgmsg(img, 'mono8')         # Convert to ROS Image msg
            drawable = self.c.handle_msg(img_msg)                   # Extract chessboard corners using ROS camera_calibration package

            if display_flag and i < n_disp_img:
                ax = plt.subplot(gs[0, 0])
                plt.imshow(img, cmap='gray')
                plt.axis('off')

                ax = plt.subplot(gs[0, 1])
                plt.imshow(drawable.scrib)
                plt.axis('off')

                plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
                fig.canvas.set_window_title('Corner Extraction (Chessboard {0})'.format(i+1))

                plt.show(block=False)
                plt.waitforbuttonpress()

        # Useful parameters
        self.d_square = square_length                             # Length of a chessboard square
        self.h_pixels, self.w_pixels = img.shape                  # Image pixel dimensions
        self.n_chessboards = len(self.c.good_corners)             # Number of examined images
        self.n_corners_y, self.n_corners_x = n_corners            # Dimensions of extracted corner grid
        self.n_corners_per_chessboard = n_corners[0]*n_corners[1]

    def undistortImages(self, A, k=np.zeros(2), n_disp_img=1e5, scale=0):
        Anew_no_k, roi = cv2.getOptimalNewCameraMatrix(A, np.zeros(4), (self.w_pixels, self.h_pixels), scale)
        mapx_no_k, mapy_no_k = cv2.initUndistortRectifyMap(A, np.zeros(4), None, Anew_no_k, (self.w_pixels, self.h_pixels), cv2.CV_16SC2)
        Anew_w_k, roi = cv2.getOptimalNewCameraMatrix(A, np.hstack([k, 0, 0]), (self.w_pixels, self.h_pixels), scale)
        mapx_w_k, mapy_w_k = cv2.initUndistortRectifyMap(A, np.hstack([k, 0, 0]), None, Anew_w_k, (self.w_pixels, self.h_pixels), cv2.CV_16SC2)

        if k[0] != 0:
            n_plots = 3
        else:
            n_plots = 2

        fig = plt.figure('Image Correction', figsize=(6*n_plots, 5))
        gs = gridspec.GridSpec(1, n_plots)
        gs.update(wspace=0.025, hspace=0.05)

        for i, file in enumerate(sorted(os.listdir(self.cal_img_path))):
            if i < n_disp_img:
                img_dist = cv2.imread(self.cal_img_path + '/' + file, 0)
                img_undist_no_k = cv2.undistort(img_dist, A, np.zeros(4), None, Anew_no_k)
                img_undist_w_k = cv2.undistort(img_dist, A, np.hstack([k, 0, 0]), None, Anew_w_k)

                ax = plt.subplot(gs[0, 0])
                ax.imshow(img_dist, cmap='gray')
                ax.axis('off')

                ax = plt.subplot(gs[0, 1])
                ax.imshow(img_undist_no_k, cmap='gray')
                ax.axis('off')

                if k[0] != 0:
                    ax = plt.subplot(gs[0, 2])
                    ax.imshow(img_undist_w_k, cmap='gray')
                    ax.axis('off')

                plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
                fig.canvas.set_window_title('Image Correction (Chessboard {0})'.format(i+1))

                plt.show(block=False)
                plt.waitforbuttonpress()

    def plotBoardPixImages(self, u_meas, v_meas, X, Y, R, t, A, n_disp_img=1e5, k=np.zeros(2)):
        # Expects X, Y, R, t to be lists of arrays, just like u_meas, v_meas

        fig = plt.figure('Chessboard Projection to Pixel Image Frame', figsize=(8, 6))
        plt.clf()

        for p in range(min(self.n_chessboards, n_disp_img)):
            plt.clf()
            ax = plt.subplot(111)
            ax.plot(u_meas[p], v_meas[p], 'r+', label='Original')

            u, v = self.transformWorld2PixImageUndist(X[p], Y[p], np.zeros(X[p].size), R[p], t[p], A)
            ax.plot(u, v, 'b+', label='Linear Intrinsic Calibration')

            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height*0.85])
            if k[0] != 0:
                u_br, v_br = self.transformWorld2PixImageDist(X[p], Y[p], np.zeros(X[p].size), R[p], t[p], A, k)
                ax.plot(u_br, v_br, 'g+', label='Radial Distortion Calibration')

            ax.axis([0, self.w_pixels, 0, self.h_pixels])
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title('Chessboard {0}'.format(p+1))
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), fontsize='medium', fancybox=True, shadow=True)

            plt.show(block=False)
            plt.waitforbuttonpress()

    def plotBoardLocations(self, X, Y, R, t, n_disp_img=1e5):
        # Expects X, U, R, t to be lists of arrays, just like u_meas, v_meas

        ind_corners = [0, self.n_corners_x-1, self.n_corners_x*self.n_corners_y-1, self.n_corners_x*(self.n_corners_y-1), ]
        s_cam = 0.02
        d_cam = 0.05
        xyz_cam = [[0, -s_cam, s_cam, s_cam, -s_cam],
                   [0, -s_cam, -s_cam, s_cam, s_cam],
                   [0, -d_cam, -d_cam, -d_cam, -d_cam]]
        ind_cam = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]]
        verts_cam = []
        for i in range(len(ind_cam)):
            verts_cam.append([zip([xyz_cam[0][j] for j in ind_cam[i]],
                                  [xyz_cam[1][j] for j in ind_cam[i]],
                                  [xyz_cam[2][j] for j in ind_cam[i]])])

        fig = plt.figure('Estimated Chessboard Locations', figsize=(12, 5))
        axim = fig.add_subplot(121)
        ax3d = fig.add_subplot(122, projection='3d')

        boards = []
        verts = []
        for p in range(self.n_chessboards):

            M = []
            W = np.column_stack((R[p], t[p]))
            for i in range(4):
                M_tld = W.dot(np.array([X[p][ind_corners[i]], Y[p][ind_corners[i]], 0, 1]))
                if np.sign(M_tld[2]) == 1:
                    Rz = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                    M_tld = Rz.dot(M_tld)
                    M_tld[2] *= -1
                M.append(M_tld[0:3])

            M = (np.array(M).T).tolist()
            verts.append([zip(M[0], M[1], M[2])])
            boards.append(Poly3DCollection(verts[p]))

        for i, file in enumerate(sorted(os.listdir(self.cal_img_path))):
            if i < n_disp_img:
                img = cv2.imread(self.cal_img_path + '/' + file, 0)
                axim.imshow(img, cmap='gray')
                axim.axis('off')

                ax3d.clear()

                for j in range(len(ind_cam)):
                    cam = Poly3DCollection(verts_cam[j])
                    cam.set_alpha(0.2)
                    cam.set_color('green')
                    ax3d.add_collection3d(cam)

                for p in range(self.n_chessboards):
                    if p == i:
                        boards[p].set_alpha(1.0)
                        boards[p].set_color('blue')
                    else:
                        boards[p].set_alpha(0.1)
                        boards[p].set_color('red')

                    ax3d.add_collection3d(boards[p])
                    ax3d.text(verts[p][0][0][0], verts[p][0][0][1], verts[p][0][0][2], '{0}'.format(p+1))
                    plt.show(block=False)

                view_max = 0.2
                ax3d.set_xlim(-view_max, view_max)
                ax3d.set_ylim(-view_max, view_max)
                ax3d.set_zlim(-2*view_max, 0)
                ax3d.set_xlabel('X axis')
                ax3d.set_ylabel('Y axis')
                ax3d.set_zlabel('Z axis')

                if i == 0:
                    ax3d.view_init(azim=90, elev=120)

                plt.tight_layout()
                fig.canvas.set_window_title('Estimated Board Locations (Chessboard {0})'.format(i+1))

                plt.show(block=False)

                raw_input('<Hit Enter To Continue>')

    def writeCalibrationYaml(self, A, k):
        self.c.intrinsics = np.array(A)
        self.c.distortion = np.hstack(([k[0], k[1]], np.zeros(3))).reshape((1, 5))
        #self.c.distortion = np.zeros(5)
        self.c.name = self.name
        self.c.R = np.eye(3)
        self.c.P = np.column_stack((np.eye(3), np.zeros(3)))
        self.c.size = [self.w_pixels, self.h_pixels]

        filename = self.name + '_calibration.yaml'
        with open(filename, 'w') as f:
            f.write(self.c.yaml())

        print('Calibration exported successfully to ' + filename)

    def getMeasuredPixImageCoord(self):
        u_meas = []
        v_meas = []
        for chessboards in self.c.good_corners:
            u_meas.append(chessboards[0][:, 0][:, 0])
            v_meas.append(self.h_pixels - chessboards[0][:, 0][:, 1])   # Flip Y-axis to traditional direction

        return u_meas, v_meas   # Lists of arrays (one per chessboard)

    def genCornerCoordinates(self, u_meas, v_meas):
        # TODO - part (i)
        # assume the lower left corner is the origin
        # [np.mod(i,self.n_corners_x)*self.d_square for i in range(self.n_corners_x*self.n_corners_y)]
        X = []
        Y = []
        for j in range(self.n_chessboards):
            x = []
            y = []
            Add_y = 0
            for i in range(self.n_corners_per_chessboard):
                if (i%self.n_corners_x) == 0:
                    Add_y = Add_y + 1
                x.append((i%self.n_corners_x) * self.d_square)
                y.append((Add_y-1)* self.d_square)
            Y.append(np.asarray(y))
            X.append(np.asarray(x))
        return X, Y

    def estimateHomography(self, u_meas, v_meas, X, Y):
        # TODO - part (ii)
        #L is a three dimensional matrix
        #It's corresponding to one chess board
        L = np.zeros((2*self.n_corners_per_chessboard,9))
        #for i in range(self.n_chessboards):
        	#Construct the L for solving the equation
    	for j in range(self.n_corners_per_chessboard):
    		L[j*2] = [X[j],Y[j],1,0,0,0,-u_meas[j]*X[j],-u_meas[j]*Y[j],-u_meas[j]]
    		L[j*2+1] = [0,0,0,X[j],Y[j],1,-v_meas[j]*X[j],-v_meas[j]*Y[j],-v_meas[j]]
    	_, _, V = np.linalg.svd(L,full_matrices=True)
    	H = V[-1]
    	H = H.reshape(3,3)
        return H

    def getCameraIntrinsics(self, H):
        # TODO - part (iii)
        #Construct V for the equation from H
        def get_H_ij(H, k, i, j):
            I = i-1
            J = j-1
        	#Because ith Column of H is hi = [h1i,h2i,h3i] Don't understand how to index this, H shape = (23,3,3)
            h = H[k][J,I]
            return h

        def get_V_ij(k, i, j):
            v = np.array([get_H_ij(H,k,i,1)*get_H_ij(H,k,j,1), 
                          get_H_ij(H,k,i,1)*get_H_ij(H,k,j,2)+get_H_ij(H,k,i,2)*get_H_ij(H,k,j,1), 
                          get_H_ij(H,k,i,2)*get_H_ij(H,k,j,2), 
                          get_H_ij(H,k,i,3)*get_H_ij(H,k,j,1)+get_H_ij(H,k,i,1)*get_H_ij(H,k,j,3),
                          get_H_ij(H,k,i,3)*get_H_ij(H,k,j,2)+get_H_ij(H,k,i,2)*get_H_ij(H,k,j,3),
                          get_H_ij(H,k,i,3)*get_H_ij(H,k,j,3)])
            return v

        V = np.zeros((2*self.n_chessboards,6))
        for i in range(self.n_chessboards):
            V[2*i] = get_V_ij(i,1,2)
            V[2*i+1] = np.subtract(get_V_ij(i,1,1), get_V_ij(i,2,2))
        _, _, b = np.linalg.svd(V,full_matrices=True)
        B = b[-1]
        B11 = B[0]
        B12 = B[1]
        B22 = B[2]
        B13 = B[3]
        B23 = B[4]
        B33 = B[5]
        print "B", B
        #Extract intrinsic parameters from B
        v0 = (B12*B13-B11*B23)/(B11*B22-B12**2)
        Lamda = B33-(B13**2+v0*(B12*B13-B11*B23))/B11
        #print Lamda
        Alpha = (Lamda/B11)**0.5
        Beta = ((Lamda*B11)/(B11*B22-B12**2))**0.5
    	Gamma = -B12*(Alpha**2)*Beta/Lamda 
    	u0 = Gamma*v0/Beta - B13*(Alpha**2)/Lamda 
        A = []
        A = np.array([[Alpha, Gamma, u0],[0, Beta, v0],[0,0,1]])
        # A = np.vstack([0, Beta, v0])
        # A = np.vstack([0,0,1])
        #print(A)
        return A

    def getExtrinsics(self, H, A):
        # TODO - part (iv)
        # solve Extrinsics for individual chess boards
        # R_list = []
        # t_list = []
        h1 = H[:,0]
        h2 = H[:,1]
        h3 = H[:,2]
        lamda = 1/np.linalg.norm((np.dot(np.linalg.inv(A),h1)))
        r1 = lamda * np.dot(np.linalg.inv(A),h1)
        r2 = lamda * np.dot(np.linalg.inv(A),h2)
        r3 = np.cross(r1,r2)
        t = lamda*np.dot(np.linalg.inv(A),h3)
        #Make sure the R is satisfying the constraints because there are so many noises.
        Q = np.vstack([r1, r2, r3])
        U, S, VT = np.linalg.svd(Q)
        R = np.matmul(U, VT)
            # R_list.append(R)
            # t_list.append(t)
        return R, t

    def transformWorld2NormImageUndist(self, X, Y, Z, R, t):
        """
        Note: The transformation functions should only process one chessboard at a time!
        This means X, Y, Z, R, t should be individual arrays
        """
        # TODO - part (v)
        # First transform the world coordinate to camera coordinate

        # First need to transform to homogeneous coordinate

        # Pw = np.array([X,Y,Z])
        # Pc = t + np.matmul(R,Pw)
        x = np.zeros(self.n_corners_per_chessboard)
        y = np.zeros(self.n_corners_per_chessboard)

        #Put R to homogenous coordinate
        R_h = np.vstack([R,[[0,0,0]]])
        print np.shape(t)
        #Put t to homogenous coordinate
        t_h = np.concatenate([t, [0]])
        #Loop through to calculate Pc
        for i in range(self.n_corners_per_chessboard):
            Pw = np.array([X[i],Y[i],Z[i],1])
            Pc_h = np.matmul(np.column_stack((R_h,t_h)),Pw)
            x[i] = Pc_h[0]/Pc_h[2]
            y[i] = Pc_h[1]/Pc_h[2]
        return x, y

    def transformWorld2PixImageUndist(self, X, Y, Z, R, t, A):
        # TODO - part (v)
        # Pw = np.array([X,Y,Z])
        # P = np.matmul(np.matmul(A,np.column_stack((R,t))),Pw)
        u = np.zeros(self.n_corners_per_chessboard)
        v = np.zeros(self.n_corners_per_chessboard)
        #Put R to homogenous coordinate
        R_h = np.vstack([R,[[0,0,0]]])
        #Put t to homogenous coordinate
        #print np.shape(t)
        t_h = np.concatenate([t, [0]])
        #print np.shape(t)
        #Put A into homogenous coordinate
        A_h = np.column_stack((A,np.transpose([0,0,0])))
        for i in range(self.n_corners_per_chessboard):
            Pw = np.array([X[i],Y[i],Z[i],1])
            P = np.matmul(np.matmul(A_h,np.column_stack((R_h,t_h))),Pw)
            u[i] = P[0]/P[2]
            v[i] = P[1]/P[2]
        return u, v

    def transformWorld2NormImageDist(self, X, Y, R, t, k):  # TODO: test
        # TODO - part (vi)
        x_br = np.zeros(self.n_corners_per_chessboard)
        y_br = np.zeros(self.n_corners_per_chessboard)
        Z = np.zeros(np.shape(X))
        A = k
        #print "HIHIHIHIHIHIH", t, np.shape(t)
        X, Y = self.transformWorld2PixImageUndist(X, Y, Z, R, t)
        for i in range(self.n_corners_per_chessboard):
            x_br[i] = X[i] + X[i] * (k[0]*(X[i]**2+Y[i]**2)+k[1]*((X[i]**2+Y[i]**2)**2))
            y_br[i] = Y[i] + Y[i] * (k[0]*(X[i]**2+Y[i]**2)+k[1]*((X[i]**2+Y[i]**2)**2))
        return x_br, y_br

    def transformWorld2PixImageDist(self, X, Y, Z, R, t, A, k):
        # TODO - part (vi)
        u_br = np.zeros(self.n_corners_per_chessboard)
        v_br = np.zeros(self.n_corners_per_chessboard)
        U, V = self.transformWorld2PixImageUndist(X, Y, Z, R, t, A)
        x, y = self.transformWorld2NormImageUndist(X, Y, R, t, A)
        u0 = A[0][2]
        v0 = A[1][2]
        for i in range(self.n_corners_per_chessboard):
            u_br[i] = U[i] + (U[i] - u0) * (k[0]*(x[i]**2+y[i]**2)+k[1]*((x[i]**2+y[i]**2)**2))
            y_br[i] = V[i] + (V[i] - v0)* (k[0]*(x[i]**2+y[i]**2)+k[1]*((x[i]**2+y[i]**2)**2))
        return u_br, v_br