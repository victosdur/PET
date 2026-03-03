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