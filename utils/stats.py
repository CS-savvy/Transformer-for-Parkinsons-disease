import statistics
import numpy as np

data = np.array([7, 5, 4, 9, 12, 45])

print("Standard Deviation of the sample is % s " % (statistics.stdev(data)))
print("Mean of the sample is % s " % (statistics.mean(data)))