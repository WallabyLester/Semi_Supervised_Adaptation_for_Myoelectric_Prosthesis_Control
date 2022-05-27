################################################## relaxed_mode ########################################################
################################################### Version 1.0.4 #######################################################
# FILE            : adaptiveTrain.py
# VERSION         : 1.0.0
# FUNCTION        : iterively compute means and covariances by class
# DEPENDENCIES    : None
# SLAVE STEP      : Add After DAQ_DATA
__author__ = 'lhargrove'
########################################################################################################################

#Import all the required modules. These are helper functions that will allow us to get variables from CAPS PC
import math
import os
if os.name == "nt":
    import pcepy.mtrx as mtrx
    import pcepy.pce as pce
    import pcepy.feat as feat
else:
    import mtrx, pce, feat
import time
import datetime
import can
import numpy as np

############################################# MAIN FUNCTION LOOP #######################################################
def run():

    # Motor enable PCE Variable
    train_status = pce.get_var('TRAIN_STATUS')      #See if the PCE is training

    #If the PCE is training, update the means and covariances, and number of points in each class
    #update a flag that tells me that I need to eventually update the classifier

    if train_status == 1:
        pce.set_var('UPDATE_CLASS',1)                  #Change the flag so we know something has been updated in training
        daq_fname = pce.get_var('DAQ_IN_FNAME')        #Get the filename of the training file
        class_num = daq_fname[-31:-27]                 #Parse the class number, date, time from the filefolder
        year_val = daq_fname[-26:-22]
        month_val = daq_fname[-22:-20]
        day_val = daq_fname[-20:-18]
        hour_val = daq_fname[-17:-15]
        minute_val = daq_fname[-15:-13]
        second_val  = daq_fname[-13:-11]
        daq_data = pce.get_var('DAQ_DATA').to_np_array()                            #Get the frame of data
        classifier_chan = pce.get_var('CLASFR_CHAN').to_np_array().astype(bool)     #Find out which channels are selected
        active_chan = classifier_chan[0:8,0]
        chans = []

        for i in range(0,8):                            #Make it dynamic for number of channels selected, assume 8 as upper
            if active_chan[i] == 1:                     #limit for now
                chans.append(i)
        class_labels = pce.get_var('CLASFR_CLAS1').to_np_array()    #The the classes to be trained
        select_chans = np.array(chans,np.float64,order='F')
        pce.set_var('SELECT_CHANS',select_chans)                    #Store this variable to use later in real-time computations
        select_chans = np.array(select_chans,np.int)
        feat_data = feat.extract(47,daq_data[select_chans,:])       #extract the feature, '47' means use TD+AR

        #We need to know if we should wipe the memory. The cue to do this is holding the button for two beeps.  This changes
        #the DVTRAIN_NUM_OF_TRAIN_SETS variable from to 2

        dvtrain_status = pce.get_var('DVTRAIN_NUM_OF_TRAIN_SETS')
        if dvtrain_status == 2:

            #If we are wiping memory, we want to retrain with only the most recent training set. Get the time and compare
            #to the folder value time

            secs = datetime.datetime(int(year_val),int(month_val), int(day_val), int(hour_val), int(minute_val), int(second_val))
            nowtime = datetime.datetime.now()
            etime = (nowtime-secs).total_seconds()

            #The first time through this loop we want to wipe the memory.  After that, we want to accumulate means and
            #covariances.  The "CLEAR_HISTORY" variable gets checked and reset to enforce this condition.

            clear_history = pce.get_var('CLEAR_HISTORY')                                #The lines below will reinitialize the means,covs, and number of points
            if clear_history == 1:                                                      #and will reset the history flag
                counter = 1;
                class_labels = pce.get_var('CLASFR_CLAS1').to_np_array()
                for i in class_labels:
                    if i < 10:
                        val = "C00" + str(int(i))
                    else:
                        val = "C0" + str(int(i))
                    meanval = "MEAN_" + val
                    covval = "COV_" + val
                    nval = "N_" + val
                    init_means = np.zeros((1,80), order = 'F')
                    init_covs = np.zeros((80,80), order = 'F')
                    pce.set_var(covval, init_covs)
                    pce.set_var(nval, 0)
                    pce.set_var(meanval, init_means)
                    counter = counter+1;
                star_rating_init = np.zeros((4,8),order= 'F')
                pce.set_var('MEAN_RATING', star_rating_init)
                pce.set_var('MEAN_PATS', 0)
                pce.set_var('STAR_RATING', star_rating_init)
                pce.set_var('STAR_PATS', 0)
                pce.set_var('CLEAR_HISTORY', 0)

            #We now need to accumulate means, covs if the data file corresponds to the most recent training session
            #if it does, then we start updating the means and covariances

            num_classes = np.shape(class_labels)
            num_classes = num_classes[0]
            if etime < (15*num_classes):                                #check to see if datafile was collected recently (15 seconds * num_classes)

                class_num =  daq_fname[-31:-27]
                meanval = 'MEAN_' + class_num
                covval = 'COV_' + class_num
                nval = 'N_' + class_num
                mean_C = pce.get_var(meanval).to_np_array()
                cov_C = pce.get_var(covval).to_np_array()
                N_C = pce.get_var(nval)
                mean_C,cov_C,N_C = updateMeanAndCov(mean_C,cov_C, N_C, feat_data)

                #the no movement class is special as we need the NM threshold and the sig quality of NM
                if class_num == 'C001':
                    pce.set_var(nval, N_C)
                    pce.set_var(meanval, mean_C)
                    pce.set_var(covval, cov_C)
                    mean_rating = checkSignals(daq_data)
                    old_rating = pce.get_var('MEAN_RATING').to_np_array()
                    mean_pats = pce.get_var('MEAN_PATS')
                    mean_pats = mean_pats+1
                    mean_rating = mean_rating+old_rating
                    pce.set_var('MEAN_RATING',mean_rating)
                    pce.set_var('MEAN_PATS',mean_pats)

                else:
                    mean_C001 = pce.get_var('MEAN_C001').to_np_array()
                    num_feats = np.shape(mean_C001)
                    nm_mavs = mean_C001[0:num_feats[0]:10]
                    sum_nm_mavs = np.sum(nm_mavs)
                    feat_mavs = feat_data[0:num_feats[0]:10]
                    sum_feat_mavs = np.sum(feat_mavs)
                    if sum_feat_mavs > sum_nm_mavs*1.1:
                        pce.set_var(nval, N_C)
                        pce.set_var(meanval, mean_C)
                        pce.set_var(covval, cov_C)

                    #Check the input signals and update its 'star_rating'
                    star_rating = checkSignals(daq_data)
                    old_rating = pce.get_var('STAR_RATING').to_np_array()
                    star_pats = pce.get_var('STAR_PATS')
                    star_pats = star_pats+1
                    star_rating = star_rating+old_rating
                    pce.set_var('STAR_RATING',star_rating)
                    pce.set_var('STAR_PATS',star_pats)

            else:
            #if it is older, don't do anything.
                pass

        #The else statement below means that DV_TRAIN is not equal to 2. In this case, we just want to update the data
        else:
            #Treat C001 'no_movement' a little bit differently than the others. This is because we need to compute the no-motion
            #threshold.

            if class_num == 'C001':
                try:
                    #The first time through, the variables MEAN_*, COV_*, and N_* may not exist.  We try to grab
                    #one of them and if it throws an exception, then we create the variables.

                    mean_C = pce.get_var('MEAN_C001').to_np_array()
                except:
                    init_means = np.zeros((1,80), order = 'F')          #if it got to here, these variables didn't exist
                    pce.set_var('MEAN_C001', init_means)                #in the PCE and needed to be created
                    init_covs = np.zeros((80,80), order = 'F')          #assume upper limits of 80 features for now.
                    pce.set_var('COV_C001', init_covs)
                    pce.set_var('N_C001', 0)                            #Class 001 is no movement. I treat it a bit differently
                                                                        #so we can construct a no-movement threshold
                    star_rating_init = np.zeros((1,8),order= 'F')
                    pce.set_var('MEAN_RATING', star_rating_init)
                    pce.set_var('MEAN_PATS', 0)
                    pce.set_var('STAR_RATING', star_rating_init)
                    pce.set_var('STAR_PATS', 0)

                mean_C = pce.get_var('MEAN_C001').to_np_array()
                cov_C = pce.get_var('COV_C001').to_np_array()
                N_C = pce.get_var('N_C001')
                mean_C,cov_C,N_C = updateMeanAndCov(mean_C,cov_C, N_C, feat_data)   #update means, covariances, and number of points
                pce.set_var('N_C001', N_C)                              #plug updated values into the PCE
                pce.set_var('MEAN_C001', mean_C)
                pce.set_var('COV_C001', cov_C)
                                #Check the input signals and update its 'star_rating'
                mean_rating = checkSignals(daq_data)
                old_rating = pce.get_var('MEAN_RATING').to_np_array()
                mean_pats = pce.get_var('MEAN_PATS')
                mean_pats = mean_pats+1
                mean_rating = mean_rating+old_rating
                pce.set_var('MEAN_RATING',mean_rating)
                pce.set_var('MEAN_PATS',mean_pats)

            else:                                                       #Else there was only 1 button push
                meanval = 'MEAN_' + class_num                           #These are for the other classes
                covval = 'COV_' + class_num
                nval = 'N_' + class_num

                try:                                                    #Construct the new variables as was done above
                    mean_C = pce.get_var(meanval).to_np_array()
                except:
                    init_means = np.zeros((1,80), order = 'F')
                    pce.set_var(meanval, init_means)
                    init_covs = np.zeros((80,80), order = 'F')
                    pce.set_var(covval, init_covs)
                    pce.set_var(nval, 0)

                mean_C = pce.get_var(meanval).to_np_array()             #Get the means, covs, and number of points
                cov_C = pce.get_var(covval).to_np_array()
                N_C = pce.get_var(nval)

                #Check the input signals and update its 'star_rating'
                star_rating = checkSignals(daq_data)
                old_rating = pce.get_var('STAR_RATING').to_np_array()
                star_pats = pce.get_var('STAR_PATS')
                star_pats = star_pats+1
                star_rating = star_rating+old_rating
                pce.set_var('STAR_RATING',star_rating)
                pce.set_var('STAR_PATS',star_pats)

                #Check to see if the data is over the NM_Thresh
                mean_C001 = pce.get_var('MEAN_C001').to_np_array()
                num_feats = np.shape(mean_C001)
                nm_mavs = mean_C001[0:num_feats[0]:10]
                sum_nm_mavs = np.sum(nm_mavs)
                feat_mavs = feat_data[0:num_feats[0]:10]
                sum_feat_mavs = np.sum(feat_mavs)
                print ("The NM Feats are :" + str(sum_nm_mavs) + "  The Feats are:" + str(sum_feat_mavs))
                if sum_feat_mavs > sum_nm_mavs*1.1:
                    mean_C,cov_C,N_C = updateMeanAndCov(mean_C,cov_C, N_C, feat_data)
                    pce.set_var(nval, N_C)
                    pce.set_var(meanval, mean_C)
                    pce.set_var(covval, cov_C)

    #Else we aren't training.  The first time through, we should update the weights and biases of the LDA WG and CG
    #and then plug it back into the PCE.  If it has already been updated, then just pass through.
    else:
        try:                                                      #if we are not training, check to see if we need to
            update_classifier = pce.get_var('UPDATE_CLASS')       #update the classifier.  Check and see if the variable
            if update_classifier == 1:                            #exists first
                class_labels = pce.get_var('CLASFR_CLAS1').to_np_array()
                (Wg,Cg) = makeLDAClassifier(class_labels)
                print Wg
                pce.set_var('WG_ACCUM',Wg)
                pce.set_var('CG_ACCUM',Cg)
                pce.set_var('UPDATE_CLASS',0)                     #update the flag so you don't recompute the LDA unless necessary
                pce.set_var('CLEAR_HISTORY',1)                    #Checking to see if I need to clear the history flag
                print "Classifier Updated"
            else:
                pass
                #print "No Classifier Update Necessary"

        #We get into this exception condition the first time if UPDATE_CLASS has not been created.  Just pass through,
        #and the UPDATE_CLASS variable will be created the first time the system gets trained.

        except:
            pass



#######################################################################################################################
# Function    : updateMeanAndCov(args)
# args        : meanMat, the previous mean: covMat: the previous covariance: N: the number of points, cur_feat: the current feature vector
# Description : This function iteratively updates means and covariance matrix based on a new feature point.
#######################################################################################################################
def updateMeanAndCov(meanMat,covMat,N,cur_feat):
    N = N+1
    ALPHA = N/(N+1)
    zero_mean_feats_old = cur_feat-meanMat                          #De-mean based on old mean value
    mean_feats = ALPHA*meanMat+(1-ALPHA)*cur_feat                   #update the mean vector
    zero_mean_feats_new = cur_feat-mean_feats                       #De-mean based on the updated mean value
    point_cov = np.dot(zero_mean_feats_old.transpose(),zero_mean_feats_new)
    point_cov = np.array(point_cov,np.float64,order='F')
    cov_updated = ALPHA*covMat+(1-ALPHA)*point_cov                  #update the covariance
    return (mean_feats,cov_updated,N)


#######################################################################################################################
# Function    : checkSignals(args)
# args        : daq_date, a frame of data
# Description : Throws flags if the signal amlpitude if very low, the signals saturate, or if there is 60 Hz noise.
#######################################################################################################################


def checkSignals(daq_data):
    num_sigs = np.shape(daq_data)
    num_sigs = num_sigs[0]
    star_rating = np.zeros((4,8))

    #Loop through the signals
    for i in range(0,num_sigs,1):
        x = daq_data[i,:]
        varData = np.std(daq_data[i,:])
        meanData = np.mean(daq_data[i,:])
        kurtData = kurt(daq_data[i,:])
        low_sigs = ([np.where( x < 1000 )])
        high_sigs = ([np.where( x > 64500 )])
        low_sigs = np.shape(low_sigs)
        low_sigs = low_sigs[2]
        high_sigs = np.shape(high_sigs)
        high_sigs = high_sigs[2]

        #The logic is:  low stand dev, and a mean of around 32,700. This corresponds to very low-amplitude signals,
        #centered around 0 V.   We'll code that as a flag in row 0
        if varData < 2000 and meanData < 34700 and meanData > 30700:
            star_rating[0,i] = 1

        #Lets look for lots of data points near the end ranges indicating that gain might be too high.  If we have
        #have lots of points near 1 or 65536, then we are likely saturating, and if the kurtosis is large, we are likely
        #saturating because of poor electrode contact. Flag in row 1
        elif (low_sigs > 5 or high_sigs > 5) and kurtData <2:
            star_rating[1,i] = 1

        #Lets look for lots of data points near the end ranges indicating that gain might be too high.  If we have
        #have lots of points near 1 or 65536, then we are likely saturating, and is smaller, then the gain is likely jus
        #set too high. Flag in row 2.
        elif (low_sigs > 5 or high_sigs > 5) and kurtData >2:
            star_rating[2,i] = 1

        #If we aren't saturating, but we have a high kurtosis, then there may be liftoff or poor contact. flag in row 3
        elif (low_sigs < 5 or high_sigs < 5) and kurtData <2:
            star_rating[3,i] = 1

        #else we are good!
        else:
            pass
    star_rating = np.array(star_rating,np.float64,order='F')
    return (star_rating)

def kurt(a):
    kurtVal = (np.sum((a - np.mean(a)) ** 4)/len(a)) / np.std(a)**4
    return(kurtVal)

#######################################################################################################################
# Function    : makeLDAClassifier(args)
# args        : class_list, the list of class labels in the classifier
# Description : Will compute the LDA weights and biases.
#######################################################################################################################

def makeLDAClassifier(class_list):
    for i in class_list:                                            #Some book-keeping to keep variables straight
        if i < 10:
            val = "C00" + str(int(i))
        else:
            val = "C0" + str(int(i))

        if i == 1:                                                  #Build pooled covariance, assumes that no-movment is always involved
            cLab = "COV_" + val
            pooled_cov = pce.get_var(cLab).to_np_array();
        else:
            cLab = "COV_" + val
            tmpVal = pce.get_var(cLab).to_np_array();
            pooled_cov = tmpVal + pooled_cov
    num_classes = np.shape(class_list)
    pooled_cov = pooled_cov/num_classes[0]
    inv_pooled_cov = np.linalg.inv(pooled_cov)                     #Find the pooled inverse covariance matrix

    for i in class_list:                                            #Some book-keeping to keep variables straight
        if i < 10:
            val = "C00" + str(int(i))
        else:
            val = "C0" + str(int(i))
        mLab = "MEAN_" + val
        mVal = pce.get_var(mLab).to_np_array();
        tmpWg = np.dot(inv_pooled_cov,mVal.T)
        tmpCg = -1/2*(mVal).dot(inv_pooled_cov).dot(mVal.T)
        if i == 1:
            Wg = tmpWg;
            Cg = tmpCg;
        else:
            Wg = np.concatenate((Wg,tmpWg),axis=1)
            Cg = np.concatenate((Cg,tmpCg),axis=1)

    Wg = np.array(Wg,np.float64,order='F')                          #Seems like this is needed to make it type compatible with the PCE
    Cg = np.array(Cg,np.float64,order='F')

    return (Wg,Cg)