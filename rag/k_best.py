import numpy as np

def get_k_best(k : int, items : list[object], scores : list[float]) -> list[object]:
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

    items = np.array(items); scores = np.array(scores)

    if (items.size != scores.size): raise ValueError("Each score must correspond to one item.")

    # 0 < k <= items.size
    k = min(k, items.size)
    k = max(k, 0)

    if items.size > 1:
        # Get the indices of the highest k elements in descending order
        # print(f"Best indices: {best_indices}")
        best_indices = np.argsort(scores)[-k:][::-1]
        best_items = [items[i] for i in best_indices]
        return best_items
    
    else:
        # No use sorting if we only have one item.
        return np.array([items]) # Wrap it in another array so that the array is 1D instead of 0D and is therefore indexable.
