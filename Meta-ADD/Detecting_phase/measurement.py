def eddr(y_pred,stream_y,delay):
    detected = 0
    for index in range(0,len(stream_y)-delay):
        if stream_y[index] == 1:
            if 1 in y_pred[index:index+delay]:
                detected += 1
    return detected/len(stream_y)




