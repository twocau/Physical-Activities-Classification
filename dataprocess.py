import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



### Create column names ###
def exportColName():
    handColName=['hand_temp', 'hand_acc16g_x', 'hand_acc16g_y', 'hand_acc16g_z', 'hand_acc6g_x', 'hand_acc6g_y', 'hand_acc6g_z', 
                 'hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z', 'hand_mag_x', 'hand_mag_y', 'hand_mag_z', 'hand_ori_0', 'hand_ori_1', 
                 'hand_ori_2', 'hand_ori_3']
    chestColName=['chest_temp', 'chest_acc16g_x', 'chest_acc16g_y', 'chest_acc16g_z', 'chest_acc6g_x', 'chest_acc6g_y', 'chest_acc6g_z', 
                  'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z', 'chest_mag_x', 'chest_mag_y', 'chest_mag_z', 'chest_ori_0', 'chest_ori_1', 
                  'chest_ori_2', 'chest_ori_3']
    ankleColName=['ankle_temp', 'ankle_acc16g_x', 'ankle_acc16g_y', 'ankle_acc16g_z', 'ankle_acc6g_x', 'ankle_acc6g_y', 'ankle_acc6g_z', 
                  'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z', 'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z', 'ankle_ori_0', 'ankle_ori_1', 
                  'ankle_ori_2', 'ankle_ori_3']
    return ['timestamp', 'activityID', 'heart_rate']+handColName+chestColName+ankleColName



### Load a single subject file & Return a dataframe ###
# subjectIndex=1,2,3...9
def loadSubject(subjectIndex):
    filename="PAMAP2_Dataset/Protocol/subject10"+str(subjectIndex)+".dat"
    col=exportColName()
    return pd.read_csv(filename, sep=' ', header=None, names=col)



### Load all subject files & Return a dataframe ###
def loadAllSubjects():
    allData=loadSubject(1)
    col=exportColName()
    for i in range(2, 10):
        filename="PAMAP2_Dataset/Protocol/subject10"+str(i)+".dat"
        tempData=pd.read_csv(filename, sep=' ', header=None, names=col)
        allData=allData.append(tempData)
          
    allData=allData.reset_index(drop=True)
    return allData



### Load listed subject files & Return a dataframe ###
# using listOfSubID=[1,2,3,4,5,6,7,8,9] is equivalent to loadAllSubjects()
def loadSubjects(listOfSubID):
    allData=loadSubject(listOfSubID[0])
    col=exportColName()
    for i in listOfSubID[1:]:
        filename="PAMAP2_Dataset/Protocol/subject10"+str(i)+".dat"
        tempData=pd.read_csv(filename, sep=' ', header=None, names=col)
        allData=allData.append(tempData)
          
    allData=allData.reset_index(drop=True)
    return allData


### Plot a pie chart of percentage of each activity ###
def plotActPercentage(data):
    activityAgg=pd.DataFrame(data, columns=['timestamp', 'activityID']).groupby('activityID').count()
    activityAgg['timestamp'].plot(kind='pie', autopct='%.2f', figsize=[6,6])
    plt.legend(loc=(1,0), labels=activityAgg.index);


### Remove activityID=0 data ###
# data is a pd.DataFrame
def removeAct0(data):
    return data.loc[lambda df: df.activityID>0, :].reset_index(drop=True)



### Remove acc6g and orientation ###
def removeAcc6gOri(data):
    colToDrop=['hand_acc6g_x', 'hand_acc6g_y', 'hand_acc6g_z', 'hand_ori_0', 'hand_ori_1', 'hand_ori_2', 'hand_ori_3',
               'chest_acc6g_x', 'chest_acc6g_y', 'chest_acc6g_z', 'chest_ori_0', 'chest_ori_1', 'chest_ori_2', 'chest_ori_3', 
               'ankle_acc6g_x', 'ankle_acc6g_y', 'ankle_acc6g_z', 'ankle_ori_0', 'ankle_ori_1', 'ankle_ori_2', 'ankle_ori_3']
    return data.drop(colToDrop, axis=1)


### Return columen names without less useful data ###
def exportEffectiveColName():
    handColName=['hand_temp', 'hand_acc16g_x', 'hand_acc16g_y', 'hand_acc16g_z', 'hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z', 
                 'hand_mag_x', 'hand_mag_y', 'hand_mag_z']
    chestColName=['chest_temp', 'chest_acc16g_x', 'chest_acc16g_y', 'chest_acc16g_z', 'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z',
                  'chest_mag_x', 'chest_mag_y', 'chest_mag_z']
    ankleColName=['ankle_temp', 'ankle_acc16g_x', 'ankle_acc16g_y', 'ankle_acc16g_z', 'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
                  'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z']
    return ['timestamp', 'activityID', 'heart_rate']+handColName+chestColName+ankleColName



### Fill NaN by (default) linear interpolation ###
def interpolateNaN(data, method='linear'):
    return data.interpolate(method=method, axis=0).ffill().bfill()



# Note: To remove the starting and ending dt sec (=dt*100 timestamp data points) of each activity, the following functions would be helpful

### Create and return a dictionary to store the dataframes of each activity without heads and tails ###
def throwAllHeadsAndTails(data, dt=10):
    actDict = exportActDict(data)
    for a in list(actDict.keys()):
        actDict[a] = throwHeadAndTail(actDict[a], dt)
    return actDict


### Create a dictionary to store the dataframes of each activity ###
def exportActDict(data):
    actID = data.activityID.unique()
    actDict = {}
    for a in actID:
        actDict[a] = data.loc[lambda df: df.activityID==a, :].reset_index(drop=True)        
    return actDict


### Return indices that show timestamp discondinuity:
# actDF is a dataframe that contains data with the same activityID
def findTimeDiscontinuity(actDF):
    time = actDF.timestamp
    tgap = np.where(abs(np.array(time[1:])-np.array(time[:-1]))>0.015)[0]
    return tgap


### Remove the head and tail of the activity from each subject ###
# dt=10 means to kill 10sec-long of data
def throwHeadAndTail(actDF, dt=10):
    # find indices that show timestamp discondinuity:
    tgap = findTimeDiscontinuity(actDF)
    n = dt*100
    
    indToDrop = np.linspace(0, n-1, n)
    for t in range(len(tgap)):
        indToDrop = np.append(indToDrop, np.linspace(tgap[t]-(n-1), tgap[t]+n, 2*n))
    # Need to take care of the case like act24. Use np.unique().
    indToDrop = np.unique(np.append(indToDrop, np.linspace(len(actDF.timestamp)-n, len(actDF.timestamp)-1, n)).astype(int))
    
    df = actDF.copy()
    df = df.drop(df.index[indToDrop]).reset_index(drop=True)
    return df




# Note: The following functions will be used for segregating spectra. Each segment will overlap the former with 1sec.

### Cut all spectra from all activities into segments ###
def chopAllSpectra(actDict, duration=5.12, withT=True):
    print('Checking discontinuity Count...')
    newActDict = {}
    for a in list(actDict.keys()):
        newActDict[a]=chopSpectrum(actDict[a], duration, withT)
        print('actID={},\t count={}/{}'.format(a, discontinuityCount(newActDict[a]), newActDict[a].shape[0]))
    return newActDict


### Check if there is any data in the 3D array whose timestamp is not continuous ###
# Should return 0 if segregate the data properly.
def discontinuityCount(db):
    t=db.shape[1]
    return np.array([((db[i][t-1][0]-db[i][0][0])>(t/100+0.1) or (db[i][t-1][0]-db[i][0][0])<(t/100-0.1)) for i in range(len(db))]).astype(int).sum()


### Return a 3D narray with many small chuncks of data ###
# duration = 5.12 sec
def chopSpectrum(actDF, duration=5.12, withT=True):
    t = int(duration*100)
    N = len(actDF)
    db = []
    if withT:
        for i in range(int(N/100)):
            if (i*100+t < N) and (actDF.timestamp[i*100+t]-actDF.timestamp[i*100]<duration+0.1) and (actDF.timestamp[i*100+t]-actDF.timestamp[i*100]>duration-0.1):
                db.append(np.array(actDF.loc[i*100:i*100+t-1, :]))
    else:
        for i in range(int(N/100)):
            delt = actDF.timestamp[i*100+t]-actDF.timestamp[i*100]
            if (i*100+t < N) and (delt<duration+0.1) and (delt>duration-0.1):
                db.append(np.array(actDF.loc[i*100:i*100+t-1, 'activityID':]))
    
    return np.array(db)



### Show the contour plot of the input spectrum segment ###
# segmentData would be a 512 x 34 array
def showContour(segmentData):
    nt, nf = segmentData.shape
    X, Y = np.meshgrid(np.linspace(0, nt-1, nt), np.linspace(0, nf-1, nf))
    Z = segmentData.transpose()
    
    fig = plt.figure(figsize=[8, 5])
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 80, cmap='YlGnBu')
    ax.view_init(50, -35)
    ax.set_xlabel('time')
    ax.set_ylabel('feature')
    ax.set_zlabel('z');


### Transform a spectrum segment matrix to a tsfresh-compatible data format ###
# replaceT=True will replace timestamp data with index
def toTsfreshFormat(segmentMatrix, replaceT=False):
    effColName = exportEffectiveColName()
    reorderColName = ['activityID']+['timestamp']+effColName[2:]
    df = pd.DataFrame(segmentMatrix, columns=effColName)
    df = df[reorderColName]
    if replaceT:
        df.timestamp = df.index
    return df