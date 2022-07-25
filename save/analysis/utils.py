def moving_average(x):
    pad_x = [x[0]] + x + [x[-1]]
    return [(pad_x[i-1] + pad_x[i] + pad_x[i+1]) / 3 for i in range(1, len(pad_x)-1)]