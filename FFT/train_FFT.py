import numpy as np 
import scipy
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.fft import fft

import librosa
import math
from sklearn.cluster import KMeans


def Divide_Frame(x,Fs,frame_size=0.03,frame_shift=0.02):
    N=len(x)
    num_sample_in_frame = math.floor(frame_size*Fs)                # so mau trong 1 khung
    num_sample_shift =math.floor(frame_shift*Fs)
    frame=[]
    i=0
    while i+num_sample_in_frame <= N :
        fr=[]
        for m in range(i,i+num_sample_in_frame):
            fr.append(x[m])
        frame.append(fr)
        i+=num_sample_shift
    return frame


def STE(frame):
    ste=[]
    for i in range (len(frame)):
        summ=0
        for m in range(len(frame[i])):
            summ+=frame[i][m]*frame[i][m]
        ste.append(summ)
    return ste/(np.max(ste))


def Mark_Threshold(ste,frame_shift=0.02):
    dd=[]
    T1=0.08
    for i in range(len(ste)):
        if ste[i]>T1 :           
            dd.append(1)
        else:
            dd.append(0)  
       
    for i in range(len(dd)-15):
        index=0
        if dd[i]==1 :
            for j in range(i,i+15): 
                if dd[j]==1:
                    index=j
            for k in range(i,index):
                dd[k]=1
    dd2=[]
    
    for i in range(len(dd)-1):
        if dd[i]==0 and dd[i+1]==1 or dd[i]==1 and dd[i+1]==0 :
            dd2.append((i+1)*frame_shift)
        if (i==0 and dd[i]==1) or (i==len(dd) and dd[i]==1):
            dd2.append(i*frame_shift)
    return dd2


def middle_frame(dd,frame,fs):
    start = int((dd[0])/0.02)
    end = int((dd[1])/0.02)
    dis=end-start
    mid_sig=[]
    for j in range(int(start+1/3*dis), int(start+2/3*dis)):
        mid_sig.append(frame[j])
    return mid_sig





def main(path):
    Fs, signal=wavfile.read(path)
    signal=signal/np.max(np.abs(signal))
    frame=Divide_Frame(signal,Fs)
    ste=STE(frame)
    dd2=Mark_Threshold(ste)
    mid_frame=np.asarray(middle_frame(dd2,frame,Fs))
    hammingWindow = np.hamming(int(0.03 * Fs))
    
    fft_arr=[]
    for i in range(len(mid_frame)):
        mid_frame[i]*=hammingWindow
        a=np.log10(np.abs(fft(mid_frame[i], 1024)))
        fft_arr.append(a[:512])
    return np.asarray(fft_arr).mean(0)
  

folder_name =['01MDA','02FVA','03MAB','04MHB','05MVB','06FTB','07FTC','08MLD','09MPD','10MSD','11MVD','12FTD','14FHH','15MMH','16FTH','17MTH','18MNK','19MXK','20MVK','21MTL','22MHL']
file_name =['a.wav','e.wav','i.wav','o.wav','u.wav']

feature_av=[]
for i in range(len(file_name)):
    avg = []
    for j in range(len(folder_name)):
        avg.append(main('../NguyenAmHuanLuyen-16k/'+folder_name[j]+'/'+file_name[i]))
    feature_av.append(np.asarray(avg).mean(0))

np.savetxt('feature_fft.txt',feature_av)



        




