import numpy as np
import DataProcess as DP
import FeatureCalculate as FC
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# scaler can be 'minmax' or 'standard'
def preprocess(subjectID, redact=1000, rescale=False, scaler='minmax'):
    subj_filename = './PAMAP2_Dataset/Protocol/subject10'+str(subjectID)+'.dat'
    col_labels = DP.col_labels
    col_sublabels = DP.col_sublabels
    HR_lim = DP.HR_lim
    HR_rest = HR_lim[subjectID][0]
    HR_max = HR_lim[subjectID][1]
    X_std = np.empty(0)
    
    #generate dataframe from the raw data
    data=pd.read_csv(subj_filename,sep=' ',names=col_labels,header=None)

    #linear interpolate missing data
    data=data.interpolate(method='linear')

    #drop columns for orientation and acc6g
    data=pd.DataFrame(data,columns=col_sublabels)

    #convert to array
    data=np.array(data)

    #normalize heart rate
    data[:,2]=DP.HR_norm(data[:,2],HR_rest,HR_max)
    
    #Rescale:
    if rescale:
        if scaler=='minmax':
            SS = MinMaxScaler(feature_range=(-1, 1), copy=True)
        else:
            SS = StandardScaler(copy=True, with_mean=True, with_std=True)
        data[:, 3:] = SS.fit_transform(data[:, 3:])
        X_std = np.copy(data[:, 3:])
        
    #computes timestamp indices where the activity changes, including 0 and l
    l=len(data)
    r=np.arange(l-1)+1
    split_ind=r[data[r,1]!=data[r-1,1]]
    split_ind=np.concatenate(([0],split_ind,[l]))

    #chop data into chunks of continuous time blocks with the same activity, also remove activity zero
    chunks=[data[split_ind[i]:split_ind[i+1]] for i in range(len(split_ind)-1) if data[split_ind[i],1]!=0]
        
    #drop the first and last n samples. Only keep redacted samples that 
    #are of sufficient length
        
    chunks=[x[redact:-(redact+1)] for x in chunks if len(x) > (2*redact)]

    return X_std, chunks


def raw_segmentation(chunks,T=512,stride=512):
    data_segmented=[]
    for chunk in chunks:
        imax=(len(chunk)-T)//stride
        for i in range(imax+1):
            arr = chunk[i*stride:i*stride+T]
            data_segmented.append(arr)
    return np.array(data_segmented)


def segmentation(chunks,T=512,stride=512):
    data_segmented=[]
    for chunk in chunks:
        imax=(len(chunk)-T)//stride
        for i in range(imax+1):
            arr = chunk[i*stride:i*stride+T]
            rawarr = arr[:, 2:].flatten('F')
            farr = feature_extraction(arr)
            totalarr = np.append(rawarr, farr)
            data_segmented.append(totalarr)
    return np.array(data_segmented)


def feature_extraction(segment):
    fc = FC.FeatureCalc()
        
    segment_df=pd.DataFrame(segment,columns=DP.col_sublabels)
    fc.load_new_ts(segment_df)
    arr=fc.calculate_features()
    return arr