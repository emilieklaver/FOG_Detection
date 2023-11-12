""" Pytorch implementation - online feature calculation
Can be used for any number of samples instead of only up to 10k"""
import pickle
from sklearn.model_selection import StratifiedKFold
from CVfunctions2 import convertTo1D, summarizeTrainValSplit, revertTo2D, generatePytorchModelInput, selectChannels, selectChannelsPyTorch, selectStudies, get_channel_min_max_torch, min_max_scaler_torch
from WindowingFunctions import renamePatientsInMultipleStudies, removeDuplicates, renameFD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import sklearn
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay, precision_recall_curve
import statsmodels.api as sm
import statsmodels as sm
import os
from keras import callbacks
from AnalysisFunctions import FOGcharacteristics
import seaborn as sn

# From colab tsai tutorial 10
from tsai.basics import *
import sktime
my_setup(sktime, sklearn)

from tsai.models.MINIROCKET_Pytorch import *
from tsai.models.utils import *
from tsai.learner import *
import time
import torch
import tensorflow as tf

# Load trainval, data, label
with open('/run/media/knf/Elements/Preprocessed data/trainvalPatients.pkl','rb') as file:
    [participants, participantCharacteristics] = pickle.load(file)
with open('/run/media/knf/Elements/Preprocessed data/trainvalXdict.pkl', 'rb') as file:
    windowedMatrix = pickle.load(file)
with open('/run/media/knf/Elements/Preprocessed data/trainvalyDict.pkl', 'rb') as file:
    FOGlabel = pickle.load(file)

print('Characteristics FOG data of complete trainval dataset')
FOGcharacteristics(FOGlabel)

# Rename patients who are in multiple studies
overviewStudieIDs = pd.read_excel('/run/media/knf/Elements/Inclusielijsten/Overview studie IDs adapted.xlsx')
overviewStudieIDs = overviewStudieIDs[overviewStudieIDs["Include in FD"] == True]
studylist = ["Pedal study", "Hololens study", "Cinoptics study", "Vibrating Socks study"]
#studylist = ["Vibrating Socks study"]

participants, participantCharacteristics = selectStudies(participants, participantCharacteristics, studylist)
participants, participantCharacteristics, overviewMultipleStudies = renamePatientsInMultipleStudies(participants, participantCharacteristics, overviewStudieIDs, studylist)
participants, participantCharacteristics = removeDuplicates(participants, participantCharacteristics)

# split trainval in k=5 groups
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
participantCharacteristics = convertTo1D(participantCharacteristics)
participants = np.asarray(participants)

t0 = time.time()
foldcounter = 0
# Select desired input data from segments and angV
# segments = ["RightLowerLeg", "RightFoot", "LeftLowerLeg", "LeftFoot"]
segments= ["Pelvis", "RightUpperLeg", "RightLowerLeg", "RightFoot", "LeftUpperLeg", "LeftLowerLeg", "LeftFoot"]
angV = False # acceleration is always used as input, angular velocity can be chosen as input

# Initialization for ROC curves
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()

for train_ix, val_ix in kfold.split(participants, participantCharacteristics): # apply cross-validation
    trainPart, valPart = participants[train_ix], participants[val_ix]
    trainPartChar, valPartChar = participantCharacteristics[train_ix], participantCharacteristics[val_ix]

    # summarize train and validation composition
    summarizeTrainValSplit(trainPartChar, valPartChar)

    # Load data of current kfold training set
    trainPartChar = revertTo2D(trainPartChar)
    valPartChar = revertTo2D(valPartChar)
    trainPart, trainPartChar = renameFD(overviewMultipleStudies, list(trainPart), trainPartChar, studylist) # Rename FD studie IDs to original names
    valPart, valPartChar = renameFD(overviewMultipleStudies, list(valPart), valPartChar, studylist)
    trainPart = np.asarray(trainPart)
    valPart = np.asarray(valPart)

    Xtrain, yTrain = generatePytorchModelInput(windowedMatrix, FOGlabel, trainPart, trainPartChar)
    Xval, yVal = generatePytorchModelInput(windowedMatrix, FOGlabel, valPart, valPartChar)
    Xtrain = selectChannels(Xtrain, segments, angV)
    Xval = selectChannels(Xval, segments, angV)

    Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[3], Xtrain.shape[2])
    Xval = Xval.reshape(Xval.shape[0], Xval.shape[3], Xval.shape[2])
    print(Xtrain.shape) # samples, channels, length (aka time or sequence steps)
    print(type(Xtrain))
    Xtrain = torch.tensor(Xtrain)
    Xval = torch.tensor(Xval)
    print(Xtrain.shape) 
    
    # Scaling down data per channel
    # x_min, x_max = get_channel_min_max_torch(Xtrain)
    # Xtrain = min_max_scaler_torch(Xtrain, x_min, x_max, -1., 1.)
    # Xval = min_max_scaler_torch(Xval, x_min, x_max, -1., 1.)
    Xtrain = Xtrain.numpy()
    Xval = Xval.numpy()

    # Data characteristics training set
    numberOfWindows = len(yTrain)
    numberOfFOG = sum(yTrain)
    print("number of windows", numberOfWindows, "number of FOG windows", numberOfFOG)

    X, y, splits = combine_split_data([Xtrain, Xval], [yTrain, yVal])

    frame = pd.DataFrame() # needed for model summary

    # tsai automatically calculates class weights

    # Using tsai/fastai, create DataLoaders for the features in X_feat.
    tfms = [None, TSClassification()]
    batch_tfms = TSStandardize(by_sample=True)
    dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)

    # model is a linear classifier Head
    model = build_ts_model(MiniRocket, dls=dls)
    print(model)

    # Drop into fastai and use it to find a good learning rate.
    learn = Learner(dls, model, metrics=accuracy) # , cbs=ShowGraph()
    learn.lr_find()
    timer.start()
    learn.fit_one_cycle(10, 3e-4)
    timer.stop()

    # Save learner
    PATH = Path('./models/MiniRocketPytorch_fold_{}.pkl'.format(foldcounter))
    PATH.parent.mkdir(parents=True, exist_ok=True)
    learn.export(PATH)

    ## EVALUATE USING TEST SET
    # evaluation on validating part of train data
    # Pass newly create features to learner
    yPredicted, _, preds = learn.get_X_preds(X[splits[1]])
    preds = preds.astype(np.float32)
    sklearn.metrics.accuracy_score(yVal, preds)

    ns_probs = [0 for _ in range(len(yVal))]
    ns_fpr, ns_tpr, _ = sklearn.metrics.roc_curve(yVal, ns_probs)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(yVal, yPredicted[:,1])  # ValueError: continuous-multioutput format is not supported

    auc = sklearn.metrics.roc_auc_score(yVal, yPredicted[:, 1])
    report = sklearn.metrics.classification_report(yVal, np.around(yPredicted[:, 1]), output_dict=True)
    # Changed yPredicted[:, 0] to yPredicted[:,1], see https://scikit-learn.org/stable/modules/model_evaluation.html #roc-auc-binary

    # Combine all information of 1 model in a table:
    model_summary = pd.DataFrame()
    model_summary = model_summary.append({
        'ROC-AUC': [(np.around(auc, 2))],
        'Sens.50': [(np.around(report['0']['recall'], 2))],
        'Spec.50': [(np.around(report['1']['recall'], 2))],
        'Validation accuracy.50': [(np.around(report['accuracy'], 2))], },
        ignore_index=True, sort=False)
    frame_train = frame.append(model_summary, ignore_index=True)
    print(frame_train)

    # Plot precision-recall curve
    precision, recall, _ = precision_recall_curve(yVal, yPredicted[:, 1])
    plt.figure()
    no_skill = (yVal == 1).sum() / len(yVal)
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    pyplot.plot(recall, precision, marker='.', label='MiniRocket')
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    pyplot.title('Precision recall curve' )
    dirname = "PrecisionRecall_MiniRocket_fold_{}.png".format(foldcounter)
    dirpath = os.path.join('/run/media/knf/Elements/', 'MiniRocket', 'PrecisionRecall', dirname)
    plt.savefig(dirpath, dpi=500)
    # show the plot
    pyplot.show()

    d = {}

    # Calculate confusionchart, accuracy, sensitivity and specificity for various thresholds
    for threshold in [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]:
        yPredictedNew = np.zeros(yPredicted[:, 1].shape)
        yPredictedNew[yPredicted[:, 1] > threshold] = 1
        conf = threshold * 2

        IdxSucces = np.where(yPredictedNew == yVal) #[:, 1])
        IdxSucces = (np.array(IdxSucces))
        NumSucces = ((IdxSucces).shape[1])
        conf = sm.stats.proportion.proportion_confint(NumSucces, yVal.shape[0], alpha=0.05, method='normal')

        d["Report " + str(threshold)] = [
            sklearn.metrics.classification_report(yVal, (yPredictedNew), output_dict=True),
            'conf int acc:', conf, 'Predicted:', yPredictedNew]

        # confusion chart
        cm = metrics.confusion_matrix(yVal, yPredictedNew)
        df_cm = pd.DataFrame(cm, range(2), range(2))
        plt.figure(figsize=(10, 7))
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        dirname = "ConfusionChart_MiniRocket_fold_{}_threshold_{}.png".format(foldcounter, threshold)
        dirpath = os.path.join('/run/media/knf/Elements/', 'MiniRocket', 'ConfusionChart', dirname)
        plt.savefig(dirpath, dpi=500)
        plt.show()

    print(d)

    # Model loss could not be made for MiniRocket

    # Save predictions and results
    dirname = "yPredicted_MiniRocket_fold_{}.pkl".format(foldcounter)
    dirpath = os.path.join('/run/media/knf/Elements/MiniRocket', 'predictions', dirname)
    with open(dirpath, 'wb') as file:
        pickle.dump(yPredicted, file)
    dirname = "yVal_MiniRocket_fold_{}.pkl".format(foldcounter)
    dirpath = os.path.join('/run/media/knf/Elements/', 'MiniRocket', 'yVal', dirname)
    with open(dirpath, 'wb') as file:
        pickle.dump(yVal, file)
    dirname = "report_MiniRocket_fold_{}.pkl".format(foldcounter)
    dirpath = os.path.join('/run/media/knf/Elements/', 'MiniRocket', 'report', dirname)
    with open(dirpath, 'wb') as file:
        pickle.dump(report, file)
    dirname = "performance_MiniRocket_fold_{}.pkl".format(foldcounter)
    dirpath = os.path.join('/run/media/knf/Elements/', 'MiniRocket', 'performance', dirname)
    with open(dirpath, 'wb') as file:
        pickle.dump(d, file)

    # Plot ROC curve for all folds and mean ROC
    viz = RocCurveDisplay.from_predictions(
        # model,
        yVal,
        yPredicted[:, 1],
        name="Fold {}".format(foldcounter),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

    dirname = "RocCurveDisplay_MiniRocket_fold_{}.pkl".format(foldcounter)
    dirpath = os.path.join('/run/media/knf/Elements/', 'MiniRocket', 'RocCurveDisplay', dirname)
    with open(dirpath, 'wb') as file:
        pickle.dump(viz, file)
        
    foldcounter += 1
    
dirname = "tprs_MiniRocket.pkl".format(foldcounter)
dirpath = os.path.join('/run/media/knf/Elements/', 'MiniRocket', 'RocCurveDisplay', dirname)
with open(dirpath, 'wb') as file:
    pickle.dump(tprs, file)
dirname = "aucs_MiniRocket.pkl".format(foldcounter)
dirpath = os.path.join('/run/media/knf/Elements/', 'MiniRocket', 'RocCurveDisplay', dirname)
with open(dirpath, 'wb') as file:
    pickle.dump(aucs, file)

# Plot ROC curve for all folds and mean ROC
ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver operating characteristic"
)
ax.legend(loc="lower right", prop={'size': 8})
ax.yaxis.label.set_size(10)
ax.xaxis.label.set_size(10)
#ax.grid(True)
# Save figure
dirname = "ROC.png"
dirpath = os.path.join('/run/media/knf/Elements/', 'MiniRocket', 'ROC', dirname)
#mng = plt.get_current_fig_manager()
#mng.full_screen_toggle()
#fig = plt.gcf()
#fig.set_size_inches((8.5, 11), forward=False) # Set image size
fig.savefig(dirpath, dpi=500)
#plt.savefig(dirpath, format="png", dpi=500)
#plt.show()

t1 = time.time()
print("duration", (t1 - t0)/3600)

print('finished')