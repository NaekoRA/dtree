class Node:
    def __init__(self):
        self.feature    = None
        self.threshold  = None
        self.children   = {}
        self.value      = None
        self.is_numeric = False
        self.value_fallback = None

def entropy(y):
    total = len(y)
    if total == 0: return 0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / total
    return -np.sum(probs * np.log2(probs + 1e-9))

def majority_class(y):
    classes, counts = np.unique(y, return_counts=True)
    return classes[np.argmax(counts)]

def information_gain(X_col, y, threshold=None, numeric=False):
    base  = entropy(y)
    total = len(y)
    if numeric and threshold is not None:
        left_mask  = X_col <= threshold
        right_mask = ~left_mask
        if left_mask.sum() == 0 or right_mask.sum() == 0: return 0
        weighted = (left_mask.sum()/total  * entropy(y[left_mask]) +
                    right_mask.sum()/total * entropy(y[right_mask]))
    else:
        weighted = 0
        for v in np.unique(X_col):
            mask      = X_col == v
            weighted += mask.sum()/total * entropy(y[mask])
    return base - weighted

def split_info(X_col, threshold=None, numeric=False):
    total = len(X_col)
    si    = 0
    if numeric and threshold is not None:
        for mask in [X_col <= threshold, X_col > threshold]:
            p = mask.sum() / total
            if p > 0: si -= p * math.log2(p)
    else:
        for v in np.unique(X_col):
            p = (X_col == v).sum() / total
            if p > 0: si -= p * math.log2(p)
    return si if si > 0 else 1

def gain_ratio(X_col, y, threshold=None, numeric=False):
    ig = information_gain(X_col, y, threshold, numeric)
    si = split_info(X_col, threshold, numeric)
    return ig / si

def best_feature(X, y, feature_names, feat_indices=None):
    best_gr     = -1
    best_feat   = None
    best_thresh = None
    best_num    = False
    indices     = feat_indices if feat_indices is not None else range(X.shape[1])
    for i in indices:
        X_col   = X[:, i]
        numeric = pd.api.types.is_numeric_dtype(X_col)
        if numeric:
            X_col      = X_col.astype(float)
            vals       = np.unique(X_col)
            thresholds = (vals[:-1] + vals[1:]) / 2
            for thresh in thresholds:
                gr = gain_ratio(X_col, y, thresh, numeric=True)
                if gr > best_gr:
                    best_gr=gr; best_feat=i
                    best_thresh=thresh; best_num=True
        else:
            gr = gain_ratio(X_col, y, numeric=False)
            if gr > best_gr:
                best_gr=gr; best_feat=i
                best_thresh=None; best_num=False
    return best_feat, best_thresh, best_num

def build_tree(X, y, feature_names, depth=0, max_depth=5, n_features=None):
    node = Node()
    node.value_fallback = majority_class(y)  
    if len(np.unique(y)) == 1:
        node.value = y[0]; return node
    if X.shape[1] == 0 or depth >= max_depth or len(y) < 2:
        node.value = majority_class(y); return node
    if n_features is not None:
        feat_indices = np.random.choice(
            X.shape[1], size=min(n_features, X.shape[1]), replace=False)
    else:
        feat_indices = None
    best_idx, thresh, numeric = best_feature(X, y, feature_names, feat_indices)
    if best_idx is None:
        node.value = majority_class(y); return node
    node.feature    = feature_names[best_idx]
    node.threshold  = thresh
    node.is_numeric = numeric
    if numeric:
        X_col      = X[:, best_idx].astype(float)
        left_mask  = X_col <= thresh
        right_mask = ~left_mask
        node.children['left']  = build_tree(
            X[left_mask],  y[left_mask],
            feature_names, depth+1, max_depth, n_features)
        node.children['right'] = build_tree(
            X[right_mask], y[right_mask],
            feature_names, depth+1, max_depth, n_features)
    else:
        X_col     = X[:, best_idx]
        rem_idx   = [i for i in range(X.shape[1]) if i != best_idx]
        new_names = [feature_names[i] for i in rem_idx]
        for v in np.unique(X_col):
            mask = X_col == v
            node.children[v] = build_tree(
                X[mask][:, rem_idx], y[mask],
                new_names, depth+1, max_depth, n_features)
    return node

def predict_one(node, feature_names, sample):
    if node.value is not None: return node.value
    feat_idx = feature_names.index(node.feature)
    if node.is_numeric:
        val = float(sample[feat_idx])
        if val <= node.threshold:
            return predict_one(node.children['left'],  feature_names, sample)
        else:
            return predict_one(node.children['right'], feature_names, sample)
    else:
        val = sample[feat_idx]
        if val not in node.children:
            return node.value_fallback
        rem_names  = [f for f in feature_names if f != node.feature]
        rem_sample = [sample[i] for i,f in enumerate(feature_names)
                      if f != node.feature]
        return predict_one(node.children[val], rem_names, rem_sample)

def predict(tree, feature_names, X):
    return np.array([predict_one(tree, feature_names, row) for row in X])

def bootstrap_sample(X, y):
    idx = np.random.randint(0, len(y), size=len(y))
    return X[idx], y[idx]

def build_forest(X, y, feature_names, n_trees=100, max_depth=5):
    n_features = max(1, int(np.sqrt(X.shape[1])))
    forest     = []
    for i in range(n_trees):
        X_s, y_s = bootstrap_sample(X, y)
        tree     = build_tree(X_s, y_s, feature_names,
                              max_depth=max_depth,
                              n_features=n_features)
        forest.append(tree)
    return forest

def predict_forest(forest, feature_names, X):
    all_preds = np.array([predict(tree, feature_names, X) for tree in forest])
    result    = []
    for col in all_preds.T:
        classes, counts = np.unique(col, return_counts=True)
        result.append(classes[np.argmax(counts)])
    return np.array(result)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

def confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    cm      = pd.DataFrame(
        np.zeros((len(classes), len(classes)), dtype=int),
        index=classes, columns=classes
    )
    for t, p in zip(y_true, y_pred):
        cm.loc[t, p] += 1
    return cm
