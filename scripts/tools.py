import numpy as np

def calculateVotingPrediction(array):
    # Function to find the most frequent element in a 1D array
    def most_frequent_element(column):
        unique_elements, counts = np.unique(column, return_counts=True)
        return unique_elements[np.argmax(counts)]
    
    # Apply the function to each column and return the result
    return np.apply_along_axis(most_frequent_element, axis=0, arr=array)

def getDataloaderTargets(dataloader):
    targets = []
    for x,y in dataloader:
        targets.extend(y.tolist())
    return np.array(targets)