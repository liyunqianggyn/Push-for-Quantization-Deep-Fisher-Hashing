from torch.autograd import Variable
import numpy as np
import torch


def extractab(test, model, classes=80):

    queryB = list([])
    queryH = list([])
    for batch_step, (data, target, _) in enumerate(test):
        var_data = Variable(data.cuda())
        H = model(var_data)
        code = torch.sign(H)
        queryB.extend(code.cpu().data.numpy())
        queryH.extend(H.cpu().data.numpy())


    queryB = np.array(queryB)
    queryH = np.array(queryH)
    return queryB, queryH

def extractab1(test, model, classes=80):

    queryB = list([])
    queryH = list([])
    for batch_step, (data, target, _) in enumerate(test):
        var_data = Variable(data.cuda())
        H= model(var_data)
        code = torch.sign(H)
        queryB.extend(code.cpu().data.numpy())
        queryH.extend(H.cpu().data.numpy())


    queryB = np.array(queryB)
    queryH = np.array(queryH)
    return queryB, queryH

def compress(train, test, model, classes=80):
    retrievalB = list([])
    retrievalL = np.ones((1, 80))
    for batch_step, (data, target, _) in enumerate(train):
        var_data = Variable(data.cuda())
        H= model(var_data)
        code = torch.sign(H)
        retrievalB.extend(code.cpu().data.numpy())
        #retrievalL.append(target)
        retrievalL = np.concatenate((retrievalL,target.numpy()), axis=0)
        
        #retrievalL = torch.cat((Variable(retrievalL),Variable(target)), 0)


    queryB = list([])
    queryL = np.ones((1, 80))
    for batch_step, (data, target, _) in enumerate(test):
        var_data = Variable(data.cuda())
        H = model(var_data)
        code = torch.sign(H)
        queryB.extend(code.cpu().data.numpy())
        #queryL.append(target)
        queryL = np.concatenate((queryL,target.numpy()), axis=0)


    retrievalB = np.array(retrievalB)  
    retrievalL = retrievalL[1:,:]
    retrievalL = np.array(retrievalL)     


    queryB = np.array(queryB)
    queryL = queryL[1:,:]
    queryL = np.array(queryL)   
    return retrievalB, retrievalL, queryB, queryL


def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    q = B2.shape[1] # max inner product value
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def calculate_map(qB, rB, queryL, retrievalL):
    """
       :param qB: {-1,+1}^{mxq} query bits
       :param rB: {-1,+1}^{nxq} retrieval bits
       :param queryL: {0,1}^{mxl} query label
       :param retrievalL: {0,1}^{nxl} retrieval label
       :return:
    """
    num_query = queryL.shape[0]
    map = 0
    for iter in range(num_query):
        # gnd : check if exists any retrieval items with same label
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        # tsum number of items with same label
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        # sort gnd by hamming dist
        hamm = calculate_hamming(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        count = np.linspace(1, tsum, tsum) # [1,2, tsum]
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        # print(map_)
        map = map + map_
    map = map / num_query
    return map


def calculate_top_map(qB, rB, queryL, retrievalL, topk):
    """
    :param qB: {-1,+1}^{mxq} query bits
    :param rB: {-1,+1}^{nxq} retrieval bits
    :param queryL: {0,1}^{mxl} query label
    :param retrievalL: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        # print(topkmap_)
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

