"""Keyword/Keyphrase Extraction using Sentence Embeddings.

Extracts keywords and keyphrases from text using:
1. N-gram generation (1-3 words)
2. Sentence embedding of document and candidates
3. Cosine similarity ranking

Usage:
    extractor = KeywordExtractor()
    keywords = extractor.extract("Your document text here...", top_k=10)
    # Returns: [{"keyword": "machine learning", "score": 0.87}, ...]
"""

import re
import string
from collections.abc import Sequence

import numpy as np


# Common English stopwords to filter out
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "once", "here", "there", "where", "why", "how", "all",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o",
    "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
    "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan",
    "shouldn", "wasn", "weren", "won", "wouldn", "i", "me", "my", "myself",
    "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "this", "that", "these",
    "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "would",
    "could", "ought", "im", "youre", "hes", "shes", "its", "were", "theyre",
    "ive", "youve", "weve", "theyve", "id", "youd", "hed", "shed", "wed",
    "theyd", "ill", "youll", "hell", "shell", "well", "theyll", "isnt",
    "arent", "wasnt", "werent", "hasnt", "havent", "hadnt", "doesnt",
    "dont", "didnt", "wont", "wouldnt", "shant", "shouldnt", "cant",
    "cannot", "couldnt", "mustnt", "lets", "thats", "whos", "whats",
    "heres", "theres", "whens", "wheres", "whys", "hows", "because",
    "as", "until", "while", "of", "also", "however", "therefore", "thus",
    "hence", "otherwise", "moreover", "furthermore", "although", "though",
}


class KeywordExtractor:
    """Extract keywords using sentence embeddings and cosine similarity.

    This extractor generates n-gram candidates from the input text,
    embeds both the document and candidates using sentence-transformers,
    and ranks candidates by their similarity to the document embedding.
    """

    def __init__(self, encoder_model=None):
        """Initialize keyword extractor.

        Args:
            encoder_model: Optional pre-loaded encoder model with encode() method.
                          If None, will use a simple TF-IDF fallback.
        """
        self.encoder_model = encoder_model

    def extract(
        self,
        text: str,
        top_k: int = 10,
        ngram_range: tuple[int, int] = (1, 3),
        diversity: float = 0.5,
    ) -> list[dict[str, float]]:
        """Extract keywords from text.

        Args:
            text: Input text to extract keywords from
            top_k: Number of keywords to return
            ngram_range: Tuple of (min_n, max_n) for n-gram generation
            diversity: MMR diversity parameter (0=no diversity, 1=max diversity)

        Returns:
            List of dicts with 'keyword' and 'score' keys, sorted by score descending
        """
        if not text or not text.strip():
            return []

        # Generate candidates
        candidates = self._generate_candidates(text, ngram_range)

        if not candidates:
            return []

        # If we have an encoder model, use embedding-based extraction
        if self.encoder_model is not None:
            return self._extract_with_embeddings(text, candidates, top_k, diversity)

        # Fallback to frequency-based extraction
        return self._extract_with_frequency(text, candidates, top_k)

    def _generate_candidates(
        self,
        text: str,
        ngram_range: tuple[int, int],
    ) -> list[str]:
        """Generate n-gram candidates from text.

        Args:
            text: Input text
            ngram_range: (min_n, max_n) for n-grams

        Returns:
            List of unique candidate phrases
        """
        # Tokenize: split on whitespace and punctuation
        # Keep only alphanumeric tokens
        tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', text.lower())

        if not tokens:
            return []

        candidates = set()
        min_n, max_n = ngram_range

        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram_tokens = tokens[i:i + n]

                # Skip if starts or ends with stopword
                if ngram_tokens[0] in STOPWORDS or ngram_tokens[-1] in STOPWORDS:
                    continue

                # Skip if all tokens are stopwords
                if all(t in STOPWORDS for t in ngram_tokens):
                    continue

                # Skip very short single tokens
                if n == 1 and len(ngram_tokens[0]) < 3:
                    continue

                ngram = " ".join(ngram_tokens)
                candidates.add(ngram)

        return list(candidates)

    def _extract_with_embeddings(
        self,
        text: str,
        candidates: list[str],
        top_k: int,
        diversity: float,
    ) -> list[dict[str, float]]:
        """Extract keywords using embedding similarity.

        Uses MMR (Maximal Marginal Relevance) for diversity.
        """
        # Embed document
        doc_embedding = self.encoder_model.encode([text])[0]

        # Embed candidates
        candidate_embeddings = self.encoder_model.encode(candidates)

        # Calculate similarities
        doc_similarities = self._cosine_similarity(
            doc_embedding.reshape(1, -1),
            candidate_embeddings
        )[0]

        # Use MMR for diverse selection
        if diversity > 0:
            selected_indices = self._mmr(
                doc_similarities,
                candidate_embeddings,
                top_k,
                diversity,
            )
        else:
            # Just take top-k by similarity
            selected_indices = np.argsort(doc_similarities)[::-1][:top_k]

        # Build results
        results = []
        for idx in selected_indices:
            results.append({
                "keyword": candidates[idx],
                "score": float(doc_similarities[idx]),
            })

        return results

    def _extract_with_frequency(
        self,
        text: str,
        candidates: list[str],
        top_k: int,
    ) -> list[dict[str, float]]:
        """Fallback extraction using term frequency.

        Simple TF-based scoring when no encoder is available.
        """
        text_lower = text.lower()

        # Count occurrences
        scores = []
        for candidate in candidates:
            count = text_lower.count(candidate)
            # Boost longer phrases
            length_boost = len(candidate.split()) * 0.1
            score = count * (1 + length_boost)
            scores.append((candidate, score))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        # Normalize scores
        max_score = scores[0][1] if scores else 1
        results = []
        for keyword, score in scores[:top_k]:
            results.append({
                "keyword": keyword,
                "score": float(score / max_score) if max_score > 0 else 0.0,
            })

        return results

    def _cosine_similarity(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity between vectors."""
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        return np.dot(a_norm, b_norm.T)

    def _mmr(
        self,
        doc_similarities: np.ndarray,
        candidate_embeddings: np.ndarray,
        top_k: int,
        diversity: float,
    ) -> list[int]:
        """Maximal Marginal Relevance for diverse keyword selection.

        Balances relevance to document with diversity among selected keywords.
        """
        # Start with most similar candidate
        selected = [int(np.argmax(doc_similarities))]
        candidates_idx = list(range(len(doc_similarities)))
        candidates_idx.remove(selected[0])

        # Iteratively select diverse keywords
        for _ in range(min(top_k - 1, len(candidates_idx))):
            if not candidates_idx:
                break

            # Calculate MMR scores
            mmr_scores = []
            for idx in candidates_idx:
                # Relevance to document
                relevance = doc_similarities[idx]

                # Max similarity to already selected
                selected_sims = self._cosine_similarity(
                    candidate_embeddings[idx].reshape(1, -1),
                    candidate_embeddings[selected]
                )
                max_sim_to_selected = np.max(selected_sims)

                # MMR score
                mmr = (1 - diversity) * relevance - diversity * max_sim_to_selected
                mmr_scores.append((idx, mmr))

            # Select best MMR candidate
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected.append(best_idx)
            candidates_idx.remove(best_idx)

        return selected
