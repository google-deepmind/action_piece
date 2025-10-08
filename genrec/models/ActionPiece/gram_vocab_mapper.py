# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""GRAM Vocabulary Mapper: Maps item features to T5 vocabulary space."""

import json
import logging
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import T5Tokenizer
from tqdm import tqdm


class GRAMVocabMapper:
    """Maps item features to T5 vocabulary tokens using TF-IDF scoring.
    
    This class implements GRAM's semantic-to-lexical translation for ActionPiece.
    Instead of using abstract feature IDs as initial tokens, we map each feature
    to the most representative T5 vocabulary token using TF-IDF scoring.
    
    Attributes:
        config: Configuration dictionary.
        t5_tokenizer: Hugging Face T5 tokenizer for vocabulary access.
        logger: Logger instance.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize GRAM vocabulary mapper.
        
        Args:
            config: Configuration dictionary with GRAM settings.
            logger: Optional logger instance.
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Load T5 tokenizer to access vocabulary (32,100 tokens for t5-small)
        t5_model_name = config.get('gram_vocab_mapping', {}).get('tokenizer', 't5-small')
        self.logger.info(f"[GRAM Vocab Mapper] Loading T5 tokenizer: {t5_model_name}")
        self.t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
        self.vocab_size = len(self.t5_tokenizer)
        
        # Configuration parameters
        self.min_tfidf_score = config.get('gram_vocab_mapping', {}).get('min_tfidf_score', 0.0)
        self.fallback_token = config.get('gram_vocab_mapping', {}).get('fallback_token', '<unk>')
        self.feature_weights = config.get('gram_vocab_mapping', {}).get('feature_weights', {})
        
        self.logger.info(f"[GRAM Vocab Mapper] T5 vocabulary size: {self.vocab_size}")
    
    def map_item2feat_to_t5_vocab(
        self, 
        item2feat: Dict[int, List[List[Any]]], 
        dataset,
        cache_path: Optional[str] = None
    ) -> Dict[int, List[List[str]]]:
        """Map item features to T5 vocabulary tokens.
        
        Args:
            item2feat: Original ActionPiece item2feat mapping.
                Format: {item_id: [[feat_0_choice_0, feat_0_choice_1, ...], 
                                   [feat_1_choice_0, ...], ...]}
            dataset: Dataset instance for accessing metadata.
            cache_path: Optional path to cache the mapping results.
            
        Returns:
            Mapped item2feat with T5 tokens.
                Format: {item_id: [[t5_token_0, t5_token_1, ...], 
                                   [t5_token_1, ...], ...]}
        """
        # Try loading from cache
        if cache_path and os.path.exists(cache_path):
            self.logger.info(f"[GRAM Vocab Mapper] Loading cached mapping from {cache_path}")
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        self.logger.info("[GRAM Vocab Mapper] Computing T5 vocabulary mapping...")
        
        # Step 1: Build corpus-level statistics for IDF computation
        self.logger.info("[GRAM Vocab Mapper] Building corpus statistics...")
        doc_texts = self._build_document_corpus(item2feat, dataset)
        idf_scores = self._compute_idf_scores(doc_texts)
        
        # Step 2: Map each item's features to T5 tokens
        self.logger.info("[GRAM Vocab Mapper] Mapping features to T5 tokens...")
        mapped_item2feat = {}
        
        for item_id in tqdm(item2feat.keys(), desc="Mapping items"):
            mapped_item2feat[item_id] = self._map_single_item(
                item_id, item2feat[item_id], dataset, doc_texts, idf_scores
            )
        
        # Save to cache
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(mapped_item2feat, f, indent=2)
            self.logger.info(f"[GRAM Vocab Mapper] Cached mapping to {cache_path}")
        
        self.logger.info(f"[GRAM Vocab Mapper] Mapped {len(mapped_item2feat)} items")
        return mapped_item2feat
    
    def _build_document_corpus(
        self, 
        item2feat: Dict[int, List[List[Any]]], 
        dataset
    ) -> Dict[int, str]:
        """Build full text representation for each item.
        
        Args:
            item2feat: Original item2feat mapping.
            dataset: Dataset instance for metadata access.
            
        Returns:
            Dictionary mapping item_id to full text.
        """
        doc_texts = {}
        
        for item_id in item2feat.keys():
            text_parts = []
            
            # Collect text from different sources
            if hasattr(dataset, 'titles') and item_id in dataset.titles:
                text_parts.append(dataset.titles[item_id])
            
            if hasattr(dataset, 'brands') and item_id in dataset.brands:
                text_parts.append(dataset.brands[item_id])
            
            if hasattr(dataset, 'categories') and item_id in dataset.categories:
                cats = dataset.categories[item_id]
                if isinstance(cats, list):
                    text_parts.extend(cats)
                else:
                    text_parts.append(str(cats))
            
            if hasattr(dataset, 'descriptions') and item_id in dataset.descriptions:
                text_parts.append(dataset.descriptions[item_id])
            
            # Combine all text
            doc_texts[item_id] = " ".join(text_parts).lower()
        
        return doc_texts
    
    def _compute_idf_scores(self, doc_texts: Dict[int, str]) -> Dict[str, float]:
        """Compute IDF (Inverse Document Frequency) scores for all tokens.
        
        Args:
            doc_texts: Dictionary mapping item_id to full text.
            
        Returns:
            Dictionary mapping token to IDF score.
        """
        n_docs = len(doc_texts)
        
        # Count document frequency for each token
        doc_freq = defaultdict(int)
        
        for doc_text in tqdm(doc_texts.values(), desc="Computing IDF"):
            # Tokenize with T5
            tokens = self.t5_tokenizer.tokenize(doc_text)
            unique_tokens = set(tokens)
            
            for token in unique_tokens:
                doc_freq[token] += 1
        
        # Compute IDF: log(N / df(t))
        idf_scores = {}
        for token, df in doc_freq.items():
            idf_scores[token] = np.log(n_docs / (df + 1))  # +1 for smoothing
        
        return idf_scores
    
    def _map_single_item(
        self,
        item_id: int,
        item_features: List[List[Any]],
        dataset,
        doc_texts: Dict[int, str],
        idf_scores: Dict[str, float]
    ) -> List[List[str]]:
        """Map a single item's features to T5 tokens.
        
        Args:
            item_id: Item identifier.
            item_features: List of feature choices for this item.
            dataset: Dataset instance.
            doc_texts: Document corpus.
            idf_scores: Precomputed IDF scores.
            
        Returns:
            Mapped features as T5 token strings.
        """
        mapped_features = []
        
        # Feature types: typically [title, brand, category, description, ...]
        feature_types = self._get_feature_types(dataset)
        
        for feat_idx, feat_choices in enumerate(item_features):
            # Get feature type name
            feat_type = feature_types[feat_idx] if feat_idx < len(feature_types) else f"feat_{feat_idx}"
            
            # Get text for this feature
            feat_text = self._get_feature_text(dataset, item_id, feat_type)
            
            # Map each choice to a T5 token
            mapped_choices = []
            for choice_idx, choice_val in enumerate(feat_choices):
                t5_token = self._select_best_t5_token(
                    feat_text, 
                    feat_type,
                    doc_texts[item_id],
                    idf_scores
                )
                mapped_choices.append(t5_token)
            
            mapped_features.append(mapped_choices)
        
        return mapped_features
    
    def _select_best_t5_token(
        self,
        feature_text: str,
        feature_type: str,
        full_doc_text: str,
        idf_scores: Dict[str, float]
    ) -> str:
        """Select the best T5 token for a feature using TF-IDF scoring.
        
        Args:
            feature_text: Text of the specific feature.
            feature_type: Type of feature (title, brand, etc.).
            full_doc_text: Full document text for context.
            idf_scores: Precomputed IDF scores.
            
        Returns:
            Best T5 token string.
        """
        if not feature_text or feature_text.strip() == "":
            return self.fallback_token
        
        # Tokenize feature text with T5
        tokens = self.t5_tokenizer.tokenize(feature_text.lower())
        
        if not tokens:
            return self.fallback_token
        
        # Compute TF (Term Frequency) within feature text
        token_counts = Counter(tokens)
        
        # Compute TF-IDF scores
        tfidf_scores = {}
        for token, tf in token_counts.items():
            # Normalize TF by feature text length
            tf_normalized = tf / len(tokens)
            
            # Get IDF score
            idf = idf_scores.get(token, 0.0)
            
            # TF-IDF score
            tfidf = tf_normalized * idf
            
            # Apply feature-type weight
            weight = self.feature_weights.get(feature_type, 1.0)
            tfidf_scores[token] = tfidf * weight
        
        # Select token with highest TF-IDF score
        if not tfidf_scores:
            return tokens[0]  # Fallback to first token
        
        best_token = max(tfidf_scores.keys(), key=lambda t: tfidf_scores[t])
        
        # Check minimum score threshold
        if tfidf_scores[best_token] < self.min_tfidf_score:
            return self.fallback_token
        
        return best_token
    
    def _get_feature_types(self, dataset) -> List[str]:
        """Infer feature types from dataset attributes.
        
        Args:
            dataset: Dataset instance.
            
        Returns:
            List of feature type names.
        """
        feature_types = []
        
        if hasattr(dataset, 'titles'):
            feature_types.append('title')
        if hasattr(dataset, 'brands'):
            feature_types.append('brand')
        if hasattr(dataset, 'categories'):
            feature_types.append('category')
        if hasattr(dataset, 'descriptions'):
            feature_types.append('description')
        
        return feature_types
    
    def _get_feature_text(self, dataset, item_id: int, feature_type: str) -> str:
        """Extract text for a specific feature type.
        
        Args:
            dataset: Dataset instance.
            item_id: Item identifier.
            feature_type: Type of feature to extract.
            
        Returns:
            Feature text as string.
        """
        if feature_type == 'title' and hasattr(dataset, 'titles'):
            return dataset.titles.get(item_id, "")
        
        elif feature_type == 'brand' and hasattr(dataset, 'brands'):
            return dataset.brands.get(item_id, "")
        
        elif feature_type == 'category' and hasattr(dataset, 'categories'):
            cats = dataset.categories.get(item_id, [])
            if isinstance(cats, list):
                return " ".join(cats)
            return str(cats)
        
        elif feature_type == 'description' and hasattr(dataset, 'descriptions'):
            return dataset.descriptions.get(item_id, "")
        
        return ""