import numpy as np
import pandas as pd

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

def saveToEXCEL(list, colNames, filename):
    df = pd.DataFrame(list)
    df.columns = colNames
    writer = pd.ExcelWriter(f'{filename}.xlsx', engine='xlsxwriter')
    df.to_excel(writer, index=False)
    writer._save()

def topResultPerformersFromEXCEL(inFilename, outFilename,attributes):
    # attributes = [(col position, threshold)]
    df = pd.read_excel(f'{inFilename}.xlsx')
    columns = df.columns.tolist()
    rows = [row.tolist() for index, row in df.iterrows()]
    results = []
    for row in rows:
        lowerThanThreshold = False
        for attribute in attributes:
            if row[attribute[0]] <= attribute[1]:
                lowerThanThreshold = True
                break
        if not lowerThanThreshold:
            results.append(row)
    saveToEXCEL(results, columns, outFilename)
