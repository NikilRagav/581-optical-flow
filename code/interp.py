import numpy as np
import pdb
#####
#   author: Xiao Zhang
#   adapted by: Nikil Ragav
#
#   Function Input 
#   v     L*M*N          the value lies on grid point which is corresponding to the meshgrid coordinates 
#   zq    M1*N1 or M2    the query points z coordinates
#   xq    M1*N1 or M2    the query points x coordinates
#   yq    M1*N1 or M2    the query points y coordinates
#         
##########
#   Function Output
#   interpv L*1			 the interpolated value at querying coordinates xq, yq, it has the same size as xq and yq.
##########

def interp2b(v, zq, yq, xq):

	if len(xq.shape) == 2 or len(yq.shape) == 2:
		dim_input = 2
		q_h = xq.shape[0]
		q_w = xq.shape[1]
		xq = xq.flatten()
		yq = yq.flatten()

	h = v.shape[0]
	w = v.shape[1]
	if xq.shape != yq.shape:
		raise 'query coordinates Xq Yq should have same shape'


	x_floor = np.floor(xq).astype(np.int32)
	y_floor = np.floor(yq).astype(np.int32)
	x_ceil 	= np.ceil(xq).astype(np.int32)
	y_ceil 	= np.ceil(yq).astype(np.int32)
	z 		= (zq).astype(np.int32)

	x_floor[x_floor<0] = 0
	y_floor[y_floor<0] = 0
	x_ceil[x_ceil<0] = 0
	y_ceil[y_ceil<0] = 0

	x_floor[x_floor>=w-1] = w-1
	y_floor[y_floor>=h-1] = h-1
	x_ceil[x_ceil>=w-1] = w-1
	y_ceil[y_ceil>=h-1] = h-1

	v1 = v[z, y_floor, x_floor]
	v2 = v[z, y_floor, x_ceil]
	v3 = v[z, y_ceil, x_floor]
	v4 = v[z, y_ceil, x_ceil]

	lh = yq - y_floor
	lw = xq - x_floor
	hh = 1 - lh
	hw = 1 - lw

	w1 = hh * hw
	w2 = hh * lw
	w3 = lh * hw
	w4 = lh * lw

	interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4

	if dim_input == 2:
		return interp_val.reshape(q_h,q_w)
	return interp_val



if __name__ == "__main__":
	print('demo of the interp2 function')
	x_mesh, y_mesh = np.meshgrid(np.arange(4),np.arange(4))
	print('x, the x meshgrid:')
	print(x_mesh)
	print('y, the y meshgrid:')
	print(y_mesh)
	v = np.arange(16).reshape(4,4)
	print('v, the value located on the coordinates above')
	print(v)
	xq_mesh, yq_mesh = np.meshgrid(np.arange(0,3.5,0.5),np.arange(0,3.5,0.5))
	print('xq_mesh, the query points x coordinates:')
	print(xq_mesh)
	print('yq_mesh, the query points y coordinates:')
	print(yq_mesh)
	interpv = interp2b(v,xq_mesh,yq_mesh)
	print('output the interpolated point at query points, here we simply upsample the original input twice')
	print(interpv)


