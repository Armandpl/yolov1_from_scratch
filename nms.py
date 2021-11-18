import torch
from torchmetrics.functional import iou

'''
    Non-maximum Suppression
    :param B: a list of proposal boxes
    :param S: corresponding confidence scores
    :param N: overlap threshold
    :return D: list of filtered proposals
'''
def nms(B, S, N):
    D = torch.tensor([])
    
    # While B is not empty 
    while B.size(dim=1) > 0:
        # get max score & store value
        index = torch.argmax(S)
        value = B[index]
        
        # store box of with best score in D and remove it from B & S
        D = torch.cat([D, value])
        B = torch.cat([B[:index], B[index+1:]])
        S = torch.cat([S[:index], S[index+1:]])
        
        # compute IOU & compare to the overlap threshold 
        indexes = []
        for (purposal, i) in B:
            score = iou(value, purposal)
            # if iou score > overlap threshold then add its index to be deleted
            if (score > N):
                indexes.append(i)
                
        # delete indexes from B & S
        temp_B = torch.tensor([])
        temp_S = torch.tensor([])
        for i in range(len(indexes)):
            prev = 0 if i == 0 else indexes[i-1]+1
            temp_B = torch.cat([temp_B, B[prev:indexes[i]]])
            temp_S = torch.cat([temp_S, S[prev:indexes[i]]])
        B = temp_B
        S = temp_S
    
    return D    
