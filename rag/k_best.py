import numpy as np

def get_k_best(k : int, items : list[object], scores : list[float]):
    """
    Given an array of items and a corresponding array of scalar item scores,
    return k items which have the highest corresponding scores in descending order.

    E.g., for item array ['apple', 'banana', 'orange', 'grapefruit', 'mango'] with scores [0.8, 0.1, -0.5, 0.2, 0.75] and k = 3,
          this function would return ['apple', 'mango', 'grapefruit'], because these items have the greatest scores.
    
    Args:
        k (int): How many items to return.
        items (list): A list of items, which can be any value.
        scores (list[float]): The scores for every item. Items with the greatest score are selected first.
    
    Returns:
        best_items (list): The list of k best items according to the highest scores.
    """

    if (len(items) != len(scores)): raise ValueError("Each score must correspond to one item.")

    # 0 < k <= len(items)
    k = min(k, len(items))
    k = max(k, 0)

    # Get the indices of the highest k elements in descending order

    best_indices = np.argsort(scores)[-k:][::-1]

    #print(f"Best indices: {best_indices}")

    best_items = [items[i] for i in best_indices[:len(items)]]

    return best_items
