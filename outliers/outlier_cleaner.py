#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    ### your code goes here
    errors = abs(predictions - net_worths)
    data = list(zip(ages.flatten(), net_worths.flatten(), errors.flatten()))
    data_sorted = sorted(data, key=lambda x: x[2])
    limit = int(len(data_sorted) * 0.9)
    cleaned_data = data_sorted[:limit]
    return cleaned_data