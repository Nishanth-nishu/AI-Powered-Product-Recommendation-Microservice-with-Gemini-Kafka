import numpy as np
import pandas as pd
import logging
from scipy import sparse

logger = logging.getLogger(__name__)

def jaccard(cooccurrence):
    """Calculate Jaccard similarity matrix"""
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        sim = cooccurrence / (diag_rows + diag_cols - cooccurrence)
    
    return np.nan_to_num(sim)

class SARModel:
    """
    Simple Algorithm for Recommendations (SAR)
    Optimized for single-node microservice usage
    """
    def __init__(
        self, 
        col_user="user_id", 
        col_item="product_id", 
        col_rating="weight", 
        col_timestamp="timestamp"
    ):
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_timestamp = col_timestamp
        
        # Internal state
        self.user_affinity = None
        self.item_similarity = None
        self.user2index = None
        self.item2index = None
        self.index2item = None
        self.n_users = 0
        self.n_items = 0

    def fit(self, df):
        """
        Train the model on interaction data
        """
        if df.empty:
            logger.warning("SAR fit called with empty DataFrame")
            return
        
        # 1. Map Users and Items to integer indices
        unique_users = df[self.col_user].unique()
        unique_items = df[self.col_item].unique()
        
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
        
        self.user2index = {u: i for i, u in enumerate(unique_users)}
        self.item2index = {i: k for k, i in enumerate(unique_items)}
        self.index2item = {k: i for i, k in self.item2index.items()} # Reverse map for retrieval
        
        # 2. Build User-Item Affinity Matrix (Sparse)
        user_idx = df[self.col_user].map(self.user2index)
        item_idx = df[self.col_item].map(self.item2index)
        
        # Create sparse matrix (Users x Items)
        self.user_affinity = sparse.coo_matrix(
            (df[self.col_rating], (user_idx, item_idx)),
            shape=(self.n_users, self.n_items)
        ).tocsr()

        # 3. Compute Item-Item Co-occurrence Matrix
        # We use binary occurrence (did they interact?) for similarity
        user_item_binary = (self.user_affinity > 0).astype(int)
        item_cooccurrence = user_item_binary.T.dot(user_item_binary)
        
        # 4. Calculate Similarity (Jaccard)
        self.item_similarity = jaccard(item_cooccurrence)
        
        logger.info(f"SAR Model trained successfully: {self.n_users} users, {self.n_items} items")

    def recommend(self, user_id, top_k=10):
        """
        Generate recommendations for a specific user
        """
        # Safety check: If model isn't trained or user is new/unknown
        if self.user2index is None or user_id not in self.user2index:
            return []
            
        user_idx = self.user2index[user_id]
        
        # Get user vector (1 x n_items)
        user_vector = self.user_affinity[user_idx, :]
        
        # Calculate scores: User History dot Similarity Matrix
        # Result is a vector of scores for all items
        scores = user_vector.dot(self.item_similarity)
        
        # Convert to dense array if sparse
        if isinstance(scores, sparse.spmatrix):
            scores = scores.toarray()
        
        scores = scores.flatten()
        
        # Remove items the user has already seen/interacted with
        seen_indices = user_vector.indices
        scores[seen_indices] = -np.inf
        
        # Get Top K indices
        # argsort sorts ascending, so we take the last top_k and reverse
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        recommendations = []
        for idx in top_indices:
            score = scores[idx]
            # Only return items with a positive score (and valid)
            if score > 0 and score != -np.inf:
                recommendations.append({
                    "product_id": self.index2item[idx],
                    "score": float(score)
                })
                
        return recommendations