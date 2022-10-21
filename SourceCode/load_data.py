import csv
import pandas as pd
import numpy as np
# 開啟 CSV 檔案
def caculate_fwd_vec(qx,qy,qz,qw):
    x = 2*qx*qz + 2*qy*qw
    y = 2*qy*qz - 2*qx*qw
    z = 1- 2*qx*qx - 2*qy*qy
    return x,y,z
def cartesian_to_eulerian(x, y, z):
    ''' 
     The (input) corresponds to (x, y, z) of a unit sphere centered at the origin (0, 0, 0)
     Returns the values (theta, phi) with:
     theta in the range 0, to 2*pi, theta can be negative, e.g. cartesian_to_eulerian(0, -1, 0) = (-pi/2, pi/2) (is equal to (3*pi/2, pi/2))
     phi in the range 0 to pi (0 being the north pole, pi being the south pole)
    '''
    r = np.sqrt(x*x+y*y+z*z)
    theta = np.arctan2(y, x)
    phi = np.arccos(z/r)
    # remainder is used to transform it in the positive range (0, 2*pi)
    theta = np.remainder(theta, 2*np.pi)
    return theta, phi

def transform_the_radians_to_original(yaw, pitch):
    '''
    Transform the yaw values from range [0, 2pi] to range [0, 1]
    Transform the pitch values from range [0, pi] to range [0, 1]
    '''
    yaw = yaw/(2*np.pi)
    pitch = pitch/np.pi
    return yaw, pitch
def load_user_data(filepath,framerate,height,width):
  '''
  Input: User trace csv filepath, video frame rate, video height and width
  Output: Array(NumofFrame,2) contain user trace. present by coordinate(x,y) of Equirectangular video in each frame
  '''
  data = pd.read_csv(filepath)
  user_trace = []
  #print(data)
  
  total_t = float(data.tail(1)["PlaybackTime"])
  print("Total_time",total_t)
  samplerate = len(data)/total_t
  print("sensor sample rate:",samplerate)
  df =  data.loc[[x for x in data.index if x%(round(samplerate/framerate))==0]]
  print(df)
  #print("index",indices)
  for i in range(len(df)):
        trace = []

        t = df.iloc[i]
        qx,qy,qz,qw = t["UnitQuaternion.x"],t["UnitQuaternion.y"],t["UnitQuaternion.z"],t["UnitQuaternion.w"]
        x,y,z = caculate_fwd_vec(qx,qy,qz,qw)
        yaw, pitch = cartesian_to_eulerian(x,y,z)
        yaw, pitch = transform_the_radians_to_original(yaw,pitch)
        trace.append(yaw*width) 
        trace.append(pitch*height)

        user_trace.append(trace)
  #print("length:",len(user_trace))
  #print(np.array(user_trace).shape)
  #print("sam:",user_trace[0])
  return np.array(user_trace)

if __name__ == '__main__':

  filepath = './Formated_Data/Experiment_1/1/video_7.csv'
  A = load_user_data(filepath,25,2560,1440)
  print("\n\n",A[:,1])
  print(A[:,1].mean())
