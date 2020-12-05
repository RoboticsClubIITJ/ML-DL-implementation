def true_positive(y_true,y_pred):
    for yt,yp in zip(y_true,y_pred):
        if yt==1 and yp==1:
            tp+=1
    return tp

