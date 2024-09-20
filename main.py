import pandas as pd
import numpy as np
import math
from collections import Counter

# Створимо вибірку даних
data = {
    'Вік': ['<30', '<30', '30-40', '>40', '>40', '>40', '30-40', '<30', '<30', '>40', '<30', '30-40', '30-40'],
    'Робота': ['Немає', 'Немає', 'Немає', 'Є', 'Є', 'Немає', 'Є', 'Є', 'Є', 'Є', 'Є', 'Є', 'Немає'],
    'Кредитна історія': ['Погана', 'Нормальна', 'Нормальна', 'Хороша', 'Погана', 'Нормальна', 'Нормальна', 'Хороша', 'Погана', 'Нормальна', 'Нормальна', 'Погана', 'Хороша'],
    'Доходи': ['Низькі', 'Середні', 'Високі', 'Високі', 'Середні', 'Низькі', 'Високі', 'Високі', 'Середні', 'Високі', 'Високі', 'Середні', 'Низькі'],
    'Видача кредиту': ['Ні', 'Так', 'Так', 'Так', 'Ні', 'Ні', 'Так', 'Так', 'Ні', 'Так', 'Так', 'Так', 'Ні']
}

df = pd.DataFrame(data)

# Функція для обчислення ентропії
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

# Функція для обчислення інформаційного виграшу
def InfoGain(data, split_attribute_name, target_name="Видача кредиту"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    Information_Gain = total_entropy - weighted_entropy
    return Information_Gain

# Функція для створення дерева рішень
def ID3(data, originaldata, features, target_attribute_name="Видача кредиту", parent_node_class=None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]
    elif len(features) == 0:
        return parent_node_class
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        item_values = [InfoGain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]
        for value in np.unique(data[best_feature]):
            value = value
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = ID3(sub_data, originaldata, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree
        return tree

# Створюємо дерево рішень
features = list(df.columns)
features.remove("Видача кредиту")
tree = ID3(df, df, features)
print(tree)
