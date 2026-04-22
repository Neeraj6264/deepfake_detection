
from __future__ import annotations
import os
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2
import dlib
import networkx as nx
from scipy.spatial import Delaunay
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ----------------------
# Config
# ----------------------
@dataclass
class Config:
    train_dir: str = r"Z:/MTECH/deepfake/paper new 2/archive/real_vs_fake/real-vs-fake/train"
    valid_dir: str = r"Z:/MTECH/deepfake/paper new 2/archive/real_vs_fake/real-vs-fake/valid"
    test_dir:  str = r"Z:/MTECH/deepfake/paper new 2/archive/real_vs_fake/real-vs-fake/test"
    
    shape_predictor_path: str = r"Z:/MTECH/deepfake/archive/shape_predictor_68_face_landmarks.dat"

    image_size: Tuple[int, int] = (64, 64)  # (W, H)
    cnn_channels: Tuple[int, int, int] = (16, 32, 64)
    num_classes: int = 2
    batch_size: int = 32
    epochs: int = 20

    # Set conservative limits while you test; increase later
    LIMIT_TRAIN: int = 5000
    LIMIT_VALID: int = 1000
    LIMIT_TEST:  int = 1000

    # Training knobs
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    dropout: float = 0.3
    seed: int = 42

CFG = Config()

# ----------------------
# Reproducibility
# ----------------------
np.random.seed(CFG.seed)
random.seed(CFG.seed)
tf.keras.utils.set_random_seed(CFG.seed)

# ----------------------
# dlib models
# ----------------------
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(CFG.shape_predictor_path)

EXPECTED_N = 68  # 68-point landmarks

# ----------------------
# Utils
# ----------------------

def normalize_adjacency(adj: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    """Return D^{-1/2} (A + I) D^{-1/2}. adj must be (N,N)."""
    A = adj.astype(np.float32)
    if add_self_loops:
        np.fill_diagonal(A, 1.0)
    deg = A.sum(axis=1)
    inv_sqrt = np.power(np.maximum(deg, 1e-6), -0.5)
    D_inv_sqrt = np.diag(inv_sqrt)
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return A_hat.astype(np.float32)


def construct_graph(landmarks: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Build normalized adjacency and node features from landmark points."""
    if landmarks is None:
        return None, None
    pts = landmarks.astype(np.float32)

    # Node features: coordinate normalization
    maxs = np.maximum(pts.max(axis=0), 1e-6)
    feats = (pts / maxs).astype(np.float32)

    try:
        tri = Delaunay(pts)
    except Exception:
        return None, None

    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                u, v = int(simplex[i]), int(simplex[j])
                if u != v:
                    edges.add((min(u, v), max(u, v)))

    N = len(pts)
    adj = np.zeros((N, N), dtype=np.float32)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0

    adj = normalize_adjacency(adj, add_self_loops=True)
    return adj, feats


def extract_landmarks_batch(images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
    results: List[Optional[np.ndarray]] = []
    for img in images:
        if img is None:
            results.append(None)
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = face_detector(gray)
        if not faces:
            results.append(None)
            continue
        # choose largest face
        face = max(faces, key=lambda r: r.width() * r.height())
        shape = landmark_predictor(gray, face)
        lm = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
        if lm.shape[0] != EXPECTED_N:
            results.append(None)
            continue
        results.append(lm)
    return results


def load_split(data_dir: str, limit: int, cnn_shape: Tuple[int, int]) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """Load (graphs, images, labels) for a split. Returns:
       graphs: list of (A_hat, X) per sample
       images: (B, H, W, 3) float32 in [0,1]
       labels: (B,) int32
    """
    samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
    for label, cls in enumerate(["real", "fake"]):
        class_path = os.path.join(data_dir, cls)
        if not os.path.isdir(class_path):
            print(f"WARN: missing dir {class_path}")
            continue
        files = sorted(os.listdir(class_path))[:limit]
        for fname in tqdm(files, desc=f"Loading {cls} from {os.path.basename(data_dir)}"):
            fpath = os.path.join(class_path, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            # Landmarks
            lm_list = extract_landmarks_batch([img])
            lm = lm_list[0]
            A_hat, X = construct_graph(lm)
            if A_hat is None or X is None or A_hat.shape[0] != EXPECTED_N:
                continue
            # CNN image (use full frame; switch to face crop if desired)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = cnn_shape[1], cnn_shape[0]
            resized = cv2.resize(rgb, (W, H)).astype(np.float32) / 255.0
            samples.append((A_hat.astype(np.float32), X.astype(np.float32), resized, label))

    if not samples:
        raise RuntimeError(f"No valid samples in {data_dir}")

    A, X, I, y = zip(*samples)
    images = np.stack(I, axis=0)
    labels = np.array(y, dtype=np.int32)
    graphs = list(zip(A, X))
    return graphs, images, labels


def format_graph_batch(graphs: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    clean = [(A, X) for (A, X) in graphs if A.shape[0] == EXPECTED_N and X.shape[0] == EXPECTED_N]
    A = np.stack([A.astype(np.float32) for A, _ in clean], axis=0)
    X = np.stack([X.astype(np.float32) for _, X in clean], axis=0)
    return A, X

# ----------------------
# Model
# ----------------------
class GraphConvLayer(layers.Layer):
    """GCN layer that expects normalized adjacency (B,N,N) and features (B,N,F)."""
    def __init__(self, output_dim: int, use_bias: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.use_bias = use_bias

    def build(self, input_shape):
        feat_dim = int(input_shape[1][-1])  # X: (B,N,F)
        self.kernel = self.add_weight(
            shape=(feat_dim, self.output_dim), initializer="glorot_uniform", trainable=True, name="kernel"
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.output_dim,), initializer="zeros", trainable=True, name="bias"
            )
        super().build(input_shape)

    def call(self, inputs):
        A_hat, X = inputs
        XW = tf.matmul(X, self.kernel)     # (B,N,out)
        AXW = tf.matmul(A_hat, XW)         # (B,N,out)
        if self.use_bias:
            AXW = AXW + self.bias
        return AXW


def build_cnn(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    c1, c2, c3 = CFG.cnn_channels
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(c1, (3,3), padding='same', activation='relu')(inp)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(c2, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(c3, (3,3), padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    return models.Model(inp, x, name="cnn_branch")


def build_hybrid_model(node_feat_dim: int, cnn_input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    N = EXPECTED_N
    A_in = layers.Input(shape=(N, N), dtype=tf.float32, name="adjacency")
    X_in = layers.Input(shape=(N, node_feat_dim), dtype=tf.float32, name="node_features")
    I_in = layers.Input(shape=cnn_input_shape, dtype=tf.float32, name="image")

    # GCN branch
    g = GraphConvLayer(64)([A_in, X_in])
    g = layers.ReLU()(g)
    g = GraphConvLayer(32)([A_in, g])
    g = layers.ReLU()(g)
    g = tf.reduce_mean(g, axis=1)  # Global average over nodes -> (B, 32)

    # CNN branch
    cnn = build_cnn(cnn_input_shape)
    c = cnn(I_in)  # (B, C)

    x = layers.Concatenate()([g, c])
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(CFG.dropout)(x)
    out = layers.Dense(CFG.num_classes, activation='softmax')(x)

    model = models.Model(inputs=[A_in, X_in, I_in], outputs=out, name="Hybrid_GCN_CNN")
    opt = tf.keras.optimizers.Adam(learning_rate=CFG.learning_rate)
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    return model

# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    W, H = CFG.image_size
    cnn_shape = (H, W, 3)

    print("\n>>> Loading data...")
    train_graphs, train_images, train_labels = load_split(CFG.train_dir, CFG.LIMIT_TRAIN, (W, H))
    valid_graphs, valid_images, valid_labels = load_split(CFG.valid_dir, CFG.LIMIT_VALID, (W, H))
    test_graphs,  test_images,  test_labels  = load_split(CFG.test_dir,  CFG.LIMIT_TEST,  (W, H))

    train_A, train_X = format_graph_batch(train_graphs)
    valid_A, valid_X = format_graph_batch(valid_graphs)
    test_A,  test_X  = format_graph_batch(test_graphs)

    node_feat_dim = train_X.shape[-1]

    # Class weights (optional)
    classes = np.unique(train_labels)
    class_weights_vals = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
    class_weights = {int(c): float(w) for c, w in zip(classes, class_weights_vals)}
    print("Class weights:", class_weights)

    print("\n>>> Building model...")
    model = build_hybrid_model(node_feat_dim, cnn_shape)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="hybrid_gcn_cnn_best.weights.h5",
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
    ]

    print("\n>>> Training...")
    history = model.fit(
        [train_A, train_X, train_images], train_labels,
        validation_data=([valid_A, valid_X, valid_images], valid_labels),
        epochs=CFG.epochs,
        batch_size=CFG.batch_size,
        class_weight=class_weights,
        verbose=1,
        callbacks=callbacks,
    )

    print("\n>>> Saving model...")
    model.save("hybrid_gcn_cnn_final.h5")

    # ----------- Plots -----------
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title('Accuracy'); plt.xlabel('epoch'); plt.ylabel('acc'); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Loss'); plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend()
    plt.tight_layout(); plt.show()

    # ----------- Evaluation -----------
    print("\n>>> Evaluating on test set...")
    test_loss, test_acc = model.evaluate([test_A, test_X, test_images], test_labels, batch_size=CFG.batch_size, verbose=1)
    print(f"Test accuracy: {test_acc:.4f}")

    probs = model.predict([test_A, test_X, test_images], batch_size=CFG.batch_size)
    preds = probs.argmax(axis=1)
    print(classification_report(test_labels, preds, target_names=['real','fake']))

    cm = confusion_matrix(test_labels, preds)
    fig, ax = plt.subplots(figsize=(4,3))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(CFG.num_classes), yticks=np.arange(CFG.num_classes),
           xticklabels=['real','fake'], yticklabels=['real','fake'],
           ylabel='True label', xlabel='Predicted label', title='Confusion Matrix')
    # text labels
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout(); plt.show()