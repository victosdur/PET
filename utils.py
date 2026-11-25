import gudhi as gd 
import numpy as np 
import tensorflow as tf 

def generate_training_data(f, x_range, num_samples):
    x = np.linspace(*x_range, num_samples)
    y = np.array([f(xi) for xi in x])
    return x, y

# function for compute PE (without tensorflow, just for test the development in tensorflow) from persistence barcode
def computePersistenceEntropy(persistentBarcode):
    l=[]
    for i in persistentBarcode:
        l.append(i[1]-i[0])
    L = sum(l)
    p=l/L
    entropia=-np.sum(p*np.log(p))
    return entropia # round(entropia,4)

# Persistent entropy calculation in TensorFlow
def persistent_entropy(D):
    persistence = tf.experimental.numpy.diff(D)
    persistence = tf.boolean_mask(tf.abs(persistence), tf.math.is_finite(persistence))
    
    P = tf.reduce_sum(persistence)
    probabilities = persistence / P
    
    # Ensures that a probability of zero will result in a logarithm of zero as well
    log_prob = tf.zeros_like(probabilities)
    mask = probabilities > 0
    log_prob = tf.tensor_scatter_nd_update(log_prob, tf.where(mask), tf.math.log(probabilities[mask]))
    
    return -tf.reduce_sum(probabilities * log_prob)

# function for lower star persistencia diagram differentiable inside tensorflow
def LowerStarsSimplex(simplextree, filtration_values, dimensions, homology_coeff_field, persistence_dim_max):
    simplextree.reset_filtration(-np.inf, 0)

    # Assign new filtration values
    for i in range(simplextree.num_vertices()):
        simplextree.assign_filtration([i], filtration_values[i])
    simplextree.make_filtration_non_decreasing()
    
    # Compute persistence diagram
    simplextree.compute_persistence(homology_coeff_field=homology_coeff_field, persistence_dim_max=persistence_dim_max)
    
    # Get vertex pairs for optimization. First, get all simplex pairs
    pairs = simplextree.lower_star_persistence_generators()
    
    L_indices = []
    for dimension in dimensions:
    
        finite_pairs = pairs[0][dimension] if len(pairs[0]) >= dimension+1 else np.empty(shape=[0,2])
        finite_pairs = np.vstack((finite_pairs,[np.argmin(filtration_values).item(),np.argmax(filtration_values).item()]))
        essential_pairs = pairs[1][dimension] if len(pairs[1]) >= dimension+1 else np.empty(shape=[0,1])
        
        finite_indices = np.array(finite_pairs.flatten(), dtype=np.int32)
        essential_indices = np.array(essential_pairs.flatten(), dtype=np.int32)

        L_indices.append((finite_indices, essential_indices))

    return L_indices

class LowerStarLayer(tf.keras.layers.Layer):
    def __init__(self, simplextree, homology_dimensions=[0], min_persistence=None, homology_coeff_field=11, persistence_dim_max=0, **kwargs):
        super().__init__(**kwargs)
        self.dimensions  = homology_dimensions
        self.simplextree = simplextree
        self.min_persistence = min_persistence if min_persistence is not None else [0. for _ in range(len(self.dimensions))]
        self.hcf = homology_coeff_field
        self.pdm = persistence_dim_max
        assert len(self.min_persistence) == len(self.dimensions)
        
    def call(self, filtration_values):
        indices = LowerStarsSimplex(self.simplextree, filtration_values.numpy(), self.dimensions, self.hcf, self.pdm)
        # Get persistence diagrams
        self.dgms = []
        for idx_dim, dimension in enumerate(self.dimensions):
            finite_dgm = tf.reshape(tf.gather(filtration_values, indices[idx_dim][0]), [-1,2])
            essential_dgm = tf.reshape(tf.gather(filtration_values, indices[idx_dim][1]), [-1,1])
            min_pers = self.min_persistence[idx_dim]
            if min_pers >= 0:
                persistent_indices = tf.where(tf.math.abs(finite_dgm[:,1]-finite_dgm[:,0]) > min_pers)
                self.dgms.append((tf.reshape(tf.gather(finite_dgm, indices=persistent_indices),[-1,2]), essential_dgm))
            else:
                self.dgms.append((finite_dgm, essential_dgm))
        return self.dgms

def generate_directions_2D(num_directions):
    """
    Generate unit directions on S^1
    """
    angles = np.linspace(0.0, 2*np.pi, num_directions, endpoint=False)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return tf.constant(directions, dtype=tf.float32), angles

def directional_height_function(x, y, direction):
    """
    Compute directional height function:
        h = <(x,y), v>
    """
    vx, vy = direction[0], direction[1]
    return vx * x + vy * y

def persistent_entropy_transform(
        x_coords,
        y_coords,
        lower_star_layer,
        num_directions=32):
    """
    Compute Persistent Entropy Transform (PET)

    Returns:
        pet_values : Tensor (num_directions,)
        angles     : sampled directions
    """

    directions, angles = generate_directions_2D(num_directions)

    pet_values = []

    for k in range(num_directions):

        direction = directions[k]

        # ---- Directional filtration ----
        filtration_values = directional_height_function(
            x_coords,
            y_coords,
            direction
        )

        # ---- Persistence diagram ----
        dgms = lower_star_layer.call(filtration_values)
        dgm = dgms[0][0]  # H0 diagram

        # ---- Persistent entropy ----
        pe = persistent_entropy(dgm)

        pet_values.append(pe)

    pet_values = tf.stack(pet_values)

    return pet_values, angles
from time import perf_counter
import pandas as pd
from scipy.special import ndtr
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from tqdm.auto import tqdm


def ecg_curve(signal, n_points):
    y = np.asarray(signal, dtype=float).reshape(-1)
    if not np.isfinite(y).all():
        raise ValueError('Non-finite signal')
    x = np.arange(len(y), dtype=float)
    x = x / (np.ptp(x) + 1e-8)
    y = (y - y.min()) / (np.ptp(y) + 1e-8)
    idx = np.linspace(0, len(y)-1, n_points).astype(int)
    return tf.constant(x[idx], dtype=tf.float32), tf.constant(y[idx], dtype=tf.float32)


def ecg_diagram(x, y, direction, layer):
    h = directional_height_function(x, y, direction)
    d = np.asarray(layer.call(h)[0][0].numpy(), dtype=np.float64).reshape(-1, 2)
    return d[np.isfinite(d).all(axis=1) & (d[:, 1] > d[:, 0])]


def ecg_summary(d, kind, direction, n_bins=100, k_max=5, image_bins=20):
    sizes = {'PE': 1, 'BC': n_bins, 'PL': n_bins*k_max, 'PI': image_bins**2}
    if not len(d):
        return np.zeros(sizes[kind], dtype=np.float32)
    b, e = d.T
    lengths = e-b
    if kind == 'PE':
        p = lengths / lengths.sum()
        return np.array([-np.dot(p, np.log(p))], dtype=np.float32)
    if kind in ('BC', 'PL'):
        grid = np.linspace(b.min(), e.max(), n_bins)
        if kind == 'BC':
            return ((b[:, None] <= grid) & (grid < e[:, None])).sum(axis=0).astype(np.float32)
        tents = np.maximum(0, np.minimum(grid[:, None]-b, e-grid[:, None]))
        top = np.sort(tents, axis=1)[:, ::-1][:, :k_max]
        return np.pad(top, ((0, 0), (0, k_max-top.shape[1]))).ravel().astype(np.float32)
    v = np.asarray(direction, dtype=float)
    lo = np.minimum(v, 0).sum()
    hi = np.maximum(v, 0).sum()
    birth_edges = np.linspace(lo, hi, image_bins+1)
    pers_edges = np.linspace(0, hi-lo, image_bins+1)
    birth_mass = np.diff(ndtr(birth_edges[None, :] - b[:, None]), axis=1)
    pers_mass = np.diff(ndtr(pers_edges[None, :] - lengths[:, None]), axis=1)
    return (birth_mass.T @ (lengths[:, None]*pers_mass)).ravel().astype(np.float32)


def ecg_features(X, layer, n_points, representation, num_directions=64):
    X = np.asarray(X)
    if representation == 'Raw':
        start = perf_counter()
        features = X.reshape(len(X), -1)
        return features, perf_counter()-start
    kinds = {'PE':'PE', 'BC':'BC', 'PL':'PL', 'PI':'PI', 'PET':'PE', 'BCT':'BC', 'PLT':'PL', 'PIT':'PI'}
    directional = representation in ('PET', 'BCT', 'PLT', 'PIT')
    directions = generate_directions_2D(num_directions)[0] if directional else tf.constant([[0., 1.]], dtype=tf.float32)
    kind = kinds[representation]
    dim = {'PE':1, 'BC':100, 'PL':500, 'PI':400}[kind]*len(directions)
    features = np.empty((len(X), dim), dtype=np.float32)
    elapsed = 0.0
    for i, signal in enumerate(tqdm(X, desc=representation)):
        start = perf_counter()
        x, y = ecg_curve(signal, n_points)
        features[i] = np.concatenate([ecg_summary(ecg_diagram(x, y, v, layer), kind, v.numpy()) for v in directions])
        elapsed += perf_counter()-start
    return features, elapsed


def ecg_models(random_state=42):
    return {
        'SVM': SVC(kernel='linear', C=1, probability=True, random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=300, random_state=random_state),
        'XGBoost': XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=random_state)
    }


def ecg_comparison(X, y, layer, n_points, X_test=None, y_test=None, n_splits=5, n_repeats=10, random_state=42, num_directions=64, output_prefix='ecg'):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError('Expected two classes')
    if X_test is not None:
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test).reshape(-1)
        if not np.isin(y_test, classes).all():
            raise ValueError('Unknown test label')
        n_train = len(X)
        X = np.concatenate([X, X_test], axis=0)
        y = np.concatenate([y, y_test])
        splits = [(np.arange(n_train), np.arange(n_train, len(X)))]
    else:
        splits = list(RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state).split(X, y))
    y = (y == classes[1]).astype(int)
    models = ecg_models(random_state)
    x0, y0 = ecg_curve(X[0], n_points)
    v0 = tf.constant([0., 1.], dtype=tf.float32)
    d0 = ecg_diagram(x0, y0, v0, layer)
    for kind in ('PE', 'BC', 'PL', 'PI'):
        ecg_summary(d0, kind, v0.numpy())
    results, timing, fold_results = [], [], []
    for representation in ('Raw', 'PE', 'BC', 'PL', 'PI', 'PET', 'BCT', 'PLT', 'PIT'):
        F, seconds = ecg_features(X, layer, n_points, representation, num_directions)
        timing.append({'Representation': representation, 'Dimension': F.shape[1], 'Extraction seconds': seconds, 'Extraction ms/sample': 1000*seconds/len(X), 'Feature matrix MiB': F.nbytes/2**20, 'Samples': len(X)})
        pd.DataFrame(timing).to_csv(f'{output_prefix}_timing.csv', index=False)
        for name, model in models.items():
            rows = []
            for fold, (tr, ts) in enumerate(tqdm(splits, desc=f'{representation} + {name}')):
                clf = clone(model)
                Ftr, Fts = F[tr], F[ts]
                start = perf_counter()
                clf.fit(Ftr, y[tr])
                fit_seconds = perf_counter()-start
                start = perf_counter()
                pred = clf.predict(Fts)
                prob = clf.predict_proba(Fts)[:, 1]
                predict_seconds = perf_counter()-start
                row = {'Accuracy': accuracy_score(y[ts], pred), 'Balanced Accuracy': balanced_accuracy_score(y[ts], pred), 'F1': f1_score(y[ts], pred), 'AUC': roc_auc_score(y[ts], prob), 'Fit seconds': fit_seconds, 'Predict seconds': predict_seconds}
                rows.append(row)
                fold_results.append({'Representation':representation, 'Model':name, 'Fold':fold, **row})
            metrics = pd.DataFrame(rows)
            result = {'Representation':representation, 'Dimension':F.shape[1], 'Model':name}
            for metric in metrics:
                result[metric+' Mean'] = metrics[metric].mean()
                result[metric+' Std'] = metrics[metric].std(ddof=0)
            results.append(result)
            pd.DataFrame(results).to_csv(f'{output_prefix}_results.csv', index=False)
            pd.DataFrame(fold_results).to_csv(f'{output_prefix}_folds.csv', index=False)
            print(result)
        del F
    return pd.DataFrame(results), pd.DataFrame(timing), pd.DataFrame(fold_results)
