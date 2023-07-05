import time
from multiprocessing import Pool
import op
import numpy as np

def signalpvd(db):
    signal = op.cs_rank(db['volume'], db['univ']) + db['close']
    print(signal)
    return signal

db2 = {
    'volume': np.asarray([[3.0, 2.0, 1.0, 5.0], [1, 2, 3, 4], [5, 6, 5, 5]]),
    'univ': np.asarray([[True, True, False, True], [True, True, True, True], [False, True, True, True]]),
    'close': np.asarray([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]])
}



if __name__ == '__main__':
    print('expected return: ', signalpvd(db2))
    results = []
    s_time = time.time()
    with Pool(4) as pool:
        for i in range(12):  # For loop used to separate keys from values.
            #print('expected input', db2)
            result = pool.apply_async(signalpvd, (db2, ))
            results.append(result.get())

        pool.close()
        pool.join()

    print(len(results))
    print('consumed time：', time.time() - s_time)