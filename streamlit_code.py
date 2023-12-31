import streamlit as st
import numpy as np
import math

st.title("ZEMO")

import streamlit as st
from PIL import Image
import io

st.title("Image Shape Detector")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    width, height = image.size
    st.write(f"Image Dimensions: {width}x{height}")
    SZ = np.array([width, height])



def zernike_order_list(order,*withneg):
    
    # Calculate the total number of input arguments
    args =zernike_order_list.__code__.co_argcount
    nargin =args + len(withneg)

    # Determine if negative orders should be included
    if nargin < 2:
       withneg = 0
    else:
       withneg = np.array(withneg)

    # Initialize lists for factorials, p-indices, and q-indices
    F = []
    for i in range(0, order+1):
        F.append(math.factorial(i))
    P = []
    Q = []

    # Generate Zernike orders based on the provided arguments
    if withneg == 1:
        for p in range(0, order+1):
            for q in range(-p, p+1):
                if (abs(p - q) % 2) == 0:
                    P.append(p)
                    Q.append(q)
    else:
        for p in range(0, order+1):
            for q in range(0, p+1):
                if (abs(p - q) % 2) == 0:
                    P.append(p)
                    Q.append(q)

    # Return the generated lists of factorials, p-indices, and q-indices
    return F,P,Q

def robust_fact_quot(X,Y):
      
      # Remove common elements from X and Y
      ca = np.unique(X)
      for k in range(len(ca)):
          i1,=np.where(X==ca[k])
          i2,=np.where(Y==ca[k])
          m=min(len(i1),len(i2))
          if m!=0:
            nn=X[i1[0]]
            X.remove(nn)
            Y.remove(nn)

      # Calculate the robust factor quotient
      R=1
      l1=len(X)
      l2=len(Y)
      for k in range(0,min(l1,l2)):
            R=R*X[k]/Y[k]
      for k in range(l2,l1):
            R=R*X[k]
      for k in range(l1,l2):
            R=R/Y[k]
      return R

def zernike_bf(SZ, order, *withneg):

  # Calculate the total number of input arguments
  args =zernike_bf.__code__.co_argcount
  nargin =args + len(withneg)

  # Determine the value of the withneg parameter
  if nargin < 3:
       withneg = 0
  else:
       withneg = 1

  # Perform Zernike order list calculation
  limitfastcomp = 50
  F,P,Q = zernike_order_list(order,withneg)
  length = len(P)
  szh = SZ / 2
  s=(1 + (2 * order), 1 + (2 * order))
  pqind = (-1) * (np.ones(s))
  src1 = (1 + order) + np.array(P)
  src2 = (1 + order) + np.array(Q)
  z = np.array([src1,src2])
  #s=np.ravel_multi_index(z-1,pqind.shape,order='F',mode=('clip'))
  z1=z[0]-1
  z2=z[1]-1
  for i in range(len(z1)):
      pqind[z1[i]][z2[i]]=i+1
  Rmns = np.zeros((1 + (2 * order), (1 + (2 * order)) + 1, 1 + (2 * order)))
  for flat in range(0 ,min(length, limitfastcomp)):
      m = P[flat]
      n = Q[flat]
      mpnh = math.floor((m + abs(n)) / 2)
      mmnh = math.floor((m - abs(n)) / 2)
      for s in range(0,mmnh+1):
        Rmns[order+m,order+n,s]=((-1) ** s) * F[m-s] / (F[s] * F[mpnh-s] * F[mmnh-s])
  for flat in range(limitfastcomp ,length):
      m = P[flat]
      n = Q[flat]
      mpnh = math.floor((m + abs(n)) / 2)
      mmnh = math.floor((m - abs(n)) / 2)
      for s in range(0,mmnh+1):
          X=[]
          Y=[]
          for i in range(1,m-s+1):
             X.append(i)
          for i in range(1,s+1):
             Y.append(i)
          for i in range(1,mpnh-s+1):
              Y.append(i)
          for i in range(1,mmnh-s+1):
              Y.append(i)
          Rmns[order+m,order+n,s]=((-1) ** s)*robust_fact_quot(X,Y)

  ZBF = np.zeros((SZ,SZ,length), dtype=complex)
  for y in range(1,SZ+1):
    for x in range(1,SZ+1):
        rho = math.sqrt(((szh - x) ** 2) + ((szh - y) ** 2))
        theta = math.atan2(szh - y, szh - x)
        if rho > szh:
              continue
        rho = rho / szh
        if theta < 0:
              theta = theta + (2 * np.pi)
        for flat in range(0,length):
            m = P[flat]
            n = Q[flat]
            R=0
            for s in range(0,int((m-abs(n))/2) +1):
              R=R+Rmns[order+m,order+n,s]*(rho**(m-2*s))
            ZBF[y-1,x-1,flat] = R*np.exp(n*theta*1j);
  pq = (P, Q)
  class Zer:
   def __init__(self, ORDER, pq, pqind, ZBF, withneg):
         
            self.maxorder = ORDER
            self.withneg = withneg
            self.orders = pq
            self.index = pqind
            self.bf = ZBF

  ZBFSTR = Zer(order, pq, pqind, ZBF, withneg)
  return ZBFSTR


def zernike_mom(I, ZBFSTR):

    # Check if the image is of square size
    if I.shape[0] != I.shape[1]:
        print("The image must be of square size!")

    # Extract necessary data from the ZBFSTR object
    bf = ZBFSTR.bf
    P  = ZBFSTR.orders[0]
    Q  = ZBFSTR.orders[1]
    Ind = ZBFSTR.index
    length = bf.shape[2]

    # Initialize an array for storing Zernike moments
    Z = np.zeros(length, dtype=complex)

    # Calculate Zernike moments
    for flat in range(0, length):
        m = P[flat]
        n = Q[flat]
        Z[flat] = ((m + 1) / np.pi) * (sum(sum((I * np.conj(bf[:,:,flat])))))

    return Z


def zernike_rec(Z, SZ, ZBFSTR, *OPTSTARTIND):

    # Extract necessary data from the ZBFSTR object
    length = len(Z)
    od = ZBFSTR.orders
    Ind = ZBFSTR.index
    bf = ZBFSTR.bf
    maxorder = ZBFSTR.maxorder
    Withneg = ZBFSTR.withneg

    # Determine the value of the OPTSTARTIND parameter
    args =zernike_rec.__code__.co_argcount
    nargin =args + len(OPTSTARTIND)

    if nargin < 4:
        if Withneg == 1:
            OPTSTARTIND = 3
        else:
            OPTSTARTIND = 2
    print(OPTSTARTIND)

    # Check if Zernike basis functions match the input vector
    if ZBFSTR.bf.shape[2] != length:
        print(ZBFSTR.bf.shape[2])
        print(length)
        print("**** ERROR *** in zernike_rec: Zernike basis functions do not match input vector!")

    # Initialize an array for the reconstructed image
    I = np.zeros((SZ,SZ), dtype=complex)

    # Reconstruct the image from Zernike moments
    if ZBFSTR.withneg:
        for i in range(OPTSTARTIND,length):
            I = I + (Z[i] * bf[:,:, i])
    else:
         for i in range(OPTSTARTIND,length):
            I = I + (Z[i] * bf[:,:, i])
            p = od[0,i]
            q = od[1,i]
            if q != 0:
                ieq = int(Ind[maxorder + p,maxorder + abs(q)]-1)
                if ieq < 1:
                    print("Invalid equivalent moment!")

                I =I + np.conj(Z[ieq]) * np.conj(bf[:,:,ieq])

    I = I.real   
    return I

Order = st.text_input("Enter an Order number:")

if st.button("Make the reconstructed image:"):
   ZBFSTR=zernike_bf(SZ[0],Order,1)
   Z = zernike_mom(np.double(image), ZBFSTR)
   I = zernike_rec(Z, SZ, ZBFSTR)
   st.image(I)
