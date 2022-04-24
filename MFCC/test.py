import numpy as np 
import scipy
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import lfilter
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
            dd2.append(i)
    return dd2


def middle_signal(dd,sig,fs):
    start = int(dd[0]*fs)
    end = int(dd[1]*fs)
    dis=end-start
    mid_sig=[]
    for j in range(int(start+1/3*dis), int(start+2/3*dis)):
        mid_sig.append(sig[j])
    return mid_sig

def preEmphasis(wave, p=0.97):
    return scipy.signal.lfilter([1.0, -p], 1, wave)

def mfcc(wave,fs,n_mfcc, win, frame_size=0.03, frame_shift=0.02  ):
    mfccs = librosa.feature.mfcc(wave, sr = fs, n_mfcc= n_mfcc, hop_length = int(frame_size*fs) - int (frame_shift*fs), n_fft = int(frame_size*fs), window=win)
    mfccs = np.average(mfccs, axis = 1)
    mfccs = mfccs.flatten()
    mfccs = mfccs.tolist()
    return mfccs

def find_feature_vector(path):
    Fs, signal=wavfile.read(path)
    signal=signal/np.max(np.abs(signal))
    frame=Divide_Frame(signal,Fs)
    ste=STE(frame)
    dd2=Mark_Threshold(ste)
    mid_signal=np.asarray(middle_signal(dd2,signal,Fs))
    hammingWindow = np.hamming(int(0.03 * Fs))
    mid_signal=preEmphasis(mid_signal)
    mfc=mfcc(mid_signal,Fs,26,hammingWindow)
    m=[]
    for i in range(len(mfc)):
        m.append( round(mfc[i], 2) )
    return np.asarray(m)

def distance(a,b):
    dis=0
    for i in range(len(a)):
        dis+=math.pow((a[i]-b[i]),2)
    return math.sqrt(dis)


def main(n_clusters):

    feature_av=np.loadtxt('k_center.txt')

    test_folder_name=['23MTL','24FTL','25MLM','27MCM','28MVN','29MHN','30FTN','32MTP','33MHP','34MQP','35MMQ','36MAQ','37MDS','38MDS','39MTS','40MHS','41MVS','42FQT','43MNT','44MTT','45MDV']
    file_name =['a.wav','e.wav','i.wav','o.wav','u.wav']
    name=['/a/','/e/','/i/','/o/','/u/']

    matrix=np.zeros((5,5))
    count = 0 
    print("KẾT QUẢ NHẬN DẠNG NGUYÊN ÂM TRÊN CÁC TÍN HIỆU KIỂM THỬ : ")
    for j in range(len(test_folder_name)):
        for i in range(len(file_name)):
            a = find_feature_vector('../NguyenAmKiemThu-16k/'+test_folder_name[j]+'/'+file_name[i])
            dis_arr=[]
            for z in range(len(feature_av)):    
                    dis_arr.append(distance(a,feature_av[z]))      
            min_ind = np.argmin(dis_arr)    
            matrix[i,(min_ind)//n_clusters] +=1
            if i == min_ind//n_clusters :
                print('NguyenAmKiemThu-16k/'+test_folder_name[j]+'/'+file_name[i]+": True")
                count+=1
            else:
                print('NguyenAmKiemThu-16k/'+test_folder_name[j]+'/'+file_name[i]+": False")
        print("-------------------")
    acc=np.round(count/((i+1)*(j+1)), 3)*100
    
    print("MA TRẬN NHẦM LẪN :")
    print("       /a/ /e/ /i/ /o/ /u/")
    print("")
    for i in range(len(matrix)):
        print(name[i],' ',matrix[i])
    print("-------------------")
    print("ACCURACY : ",acc," %")

main(3)