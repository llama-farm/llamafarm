"""Keyword extraction implementations using RAKE, YAKE, and TF-IDF."""

import re
from collections import Counter, defaultdict
from typing import Dict, Any, List, Optional
import logging
import math

from .base import BaseExtractor
from core.base import Document

logger = logging.getLogger(__name__)


class RAKEExtractor(BaseExtractor):
    """
    RAKE (Rapid Automatic Keyword Extraction) implementation.
    
    Extracts keywords and phrases without requiring external models.
    Works by identifying stop words as delimiters and scoring remaining phrases.
    """
    
    def __init__(self, name: str = "RAKE", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        
        # Configuration with defaults
        self.min_length = self.config.get("min_length", 1)
        self.max_length = self.config.get("max_length", 4)
        self.max_keywords = self.config.get("max_keywords", 10)
        self.min_frequency = self.config.get("min_frequency", 1)
        
        # Default English stop words
        self.stop_words = set(self.config.get("stop_words", self._get_default_stop_words()))
        
        # Punctuation patterns
        self.punctuation_pattern = re.compile(r'[^\w\s]')
        self.number_pattern = re.compile(r'\b\d+\b')
    
    def _get_default_stop_words(self) -> List[str]:
        """Get default English stop words."""
        return [
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if', 'no',
            'about', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her',
            'would', 'make', 'like', 'into', 'him', 'two', 'more', 'very', 'after',
            'first', 'than', 'any', 'his', 'during', 'because'
        ]
    
    def extract(self, documents: List[Document]) -> List[Document]:
        """Extract keywords using RAKE algorithm."""
        for doc in documents:
            try:
                keywords = self._extract_rake_keywords(doc.content)
                
                # Add to metadata
                if "extractors" not in doc.metadata:
                    doc.metadata["extractors"] = {}
                
                doc.metadata["extractors"]["rake_keywords"] = [
                    {"phrase": phrase, "score": score} 
                    for phrase, score in keywords
                ]
                doc.metadata["rake_keywords"] = [phrase for phrase, _ in keywords]
                
                self.logger.debug(f"Extracted {len(keywords)} RAKE keywords from document {doc.id}")
                
            except Exception as e:
                self.logger.error(f"RAKE extraction failed for document {doc.id}: {e}")
        
        return documents
    
    def _extract_rake_keywords(self, text: str) -> List[tuple]:
        """Extract keywords using RAKE algorithm."""
        # Preprocess text
        text = text.lower()
        text = self.punctuation_pattern.sub(' ', text)
        text = self.number_pattern.sub(' ', text)
        
        # Split into sentences
        sentences = re.split(r'[.!?;]', text)
        
        # Extract phrases
        phrases = []
        for sentence in sentences:
            words = sentence.split()
            phrase = []
            
            for word in words:
                if word in self.stop_words or len(word) < 2:
                    if phrase:
                        phrases.append(' '.join(phrase))
                        phrase = []
                else:
                    phrase.append(word)
            
            if phrase:
                phrases.append(' '.join(phrase))
        
        # Filter by length
        phrases = [p for p in phrases if self.min_length <= len(p.split()) <= self.max_length]
        
        # Calculate word scores
        word_freq = Counter()
        word_degree = defaultdict(int)
        
        for phrase in phrases:
            words = phrase.split()
            word_freq.update(words)
            
            for word in words:
                word_degree[word] += len(words) - 1
        
        # Calculate word scores (degree/frequency)
        word_scores = {}
        for word in word_freq:
            word_scores[word] = (word_degree[word] + word_freq[word]) / word_freq[word]
        
        # Calculate phrase scores
        phrase_scores = []
        for phrase in phrases:
            words = phrase.split()
            score = sum(word_scores[word] for word in words)
            phrase_scores.append((phrase, score))
        
        # Sort and filter
        phrase_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Remove duplicates and apply frequency filter
        seen = set()
        filtered_phrases = []
        for phrase, score in phrase_scores:
            if phrase not in seen and phrases.count(phrase) >= self.min_frequency:
                seen.add(phrase)
                filtered_phrases.append((phrase, score))
        
        return filtered_phrases[:self.max_keywords]
    
    def get_dependencies(self) -> List[str]:
        """RAKE has no external dependencies."""
        return []


class YAKEExtractor(BaseExtractor):
    """
    YAKE (Yet Another Keyword Extractor) implementation.
    
    More sophisticated than RAKE, considers position, frequency, and context.
    Lower scores indicate more important keywords.
    """
    
    def __init__(self, name: str = "YAKE", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        
        # Configuration
        self.lan = self.config.get("language", "en")
        self.n = self.config.get("max_ngram_size", 3)
        self.dedupLim = self.config.get("deduplication_threshold", 0.9)
        self.top = self.config.get("max_keywords", 10)
        self.windowSize = self.config.get("window_size", 1)
        
        # Stop words
        self.stop_words = set(self.config.get("stop_words", self._get_default_stop_words()))
    
    def _get_default_stop_words(self) -> List[str]:
        """Get default English stop words for YAKE."""
        return [
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'would', 'could', 'should', 'may', 'might',
            'can', 'cannot', 'this', 'these', 'those', 'they', 'them', 'their',
            'have', 'had', 'do', 'does', 'did', 'get', 'got', 'go', 'went', 'come',
            'came', 'see', 'saw', 'know', 'knew', 'think', 'thought', 'say', 'said'
        ]
    
    def extract(self, documents: List[Document]) -> List[Document]:
        """Extract keywords using YAKE algorithm."""
        for doc in documents:
            try:
                keywords = self._extract_yake_keywords(doc.content)
                
                # Add to metadata
                if "extractors" not in doc.metadata:
                    doc.metadata["extractors"] = {}
                
                doc.metadata["extractors"]["yake_keywords"] = [
                    {"phrase": phrase, "score": score} 
                    for phrase, score in keywords
                ]
                doc.metadata["yake_keywords"] = [phrase for phrase, _ in keywords]
                
                self.logger.debug(f"Extracted {len(keywords)} YAKE keywords from document {doc.id}")
                
            except Exception as e:
                self.logger.error(f"YAKE extraction failed for document {doc.id}: {e}")
        
        return documents
    
    def _extract_yake_keywords(self, text: str) -> List[tuple]:
        """Extract keywords using YAKE algorithm."""
        # Tokenize and clean
        words = self._tokenize_text(text)
        
        if len(words) < self.n:
            return []
        
        # Calculate word features
        word_features = self._calculate_word_features(words)
        
        # Generate candidates
        candidates = self._generate_candidates(words)
        
        # Score candidates
        scored_candidates = []
        for candidate in candidates:
            score = self._score_candidate(candidate, word_features)
            scored_candidates.append((candidate, score))
        
        # Sort by score (lower is better for YAKE)
        scored_candidates.sort(key=lambda x: x[1])
        
        # Deduplicate
        final_candidates = self._deduplicate_candidates(scored_candidates)
        
        return final_candidates[:self.top]
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple tokenization
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Filter out stop words and short words
        return [word for word in words if word not in self.stop_words and len(word) > 2]
    
    def _calculate_word_features(self, words: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate features for each word."""
        features = {}
        
        for i, word in enumerate(words):
            if word not in features:
                features[word] = {
                    'tf': 0,
                    'case': 0,
                    'position': 0,
                    'frequency': 0
                }
            
            # Term frequency
            features[word]['tf'] += 1
            
            # Position feature (earlier positions get higher scores)
            position_score = math.log(len(words) / (i + 1))
            features[word]['position'] = max(features[word]['position'], position_score)
        
        # Normalize frequencies
        total_words = len(words)
        for word in features:
            features[word]['frequency'] = features[word]['tf'] / total_words
        
        return features
    
    def _generate_candidates(self, words: List[str]) -> List[str]:
        """Generate candidate phrases."""
        candidates = set()
        
        # Generate n-grams
        for i in range(len(words)):
            for j in range(1, min(self.n + 1, len(words) - i + 1)):
                candidate = ' '.join(words[i:i + j])
                candidates.add(candidate)
        
        return list(candidates)
    
    def _score_candidate(self, candidate: str, word_features: Dict[str, Dict[str, float]]) -> float:
        """Score a candidate phrase."""
        words = candidate.split()
        
        if not words:
            return float('inf')
        
        # YAKE scoring formula (simplified)
        tf = sum(word_features.get(word, {}).get('tf', 0) for word in words)
        position = sum(word_features.get(word, {}).get('position', 0) for word in words) / len(words)
        
        # Lower scores are better in YAKE
        score = tf / (position + 1)
        
        return score
    
    def _deduplicate_candidates(self, candidates: List[tuple]) -> List[tuple]:
        """Remove similar candidates."""
        final_candidates = []
        
        for candidate, score in candidates:
            is_duplicate = False
            
            for existing_candidate, _ in final_candidates:
                # Simple similarity check
                similarity = self._calculate_similarity(candidate, existing_candidate)
                if similarity > self.dedupLim:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_candidates.append((candidate, score))
        
        return final_candidates
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_dependencies(self) -> List[str]:
        """YAKE has no external dependencies."""
        return []


class TFIDFExtractor(BaseExtractor):
    """
    TF-IDF (Term Frequency-Inverse Document Frequency) extractor.
    
    Requires a collection of documents to calculate IDF.
    Best for finding unique terms in documents relative to a corpus.
    """
    
    def __init__(self, name: str = "TFIDF", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        
        # Configuration
        self.max_features = self.config.get("max_features", 10)
        self.min_df = self.config.get("min_df", 1)
        self.max_df = self.config.get("max_df", 0.95)
        self.ngram_range = tuple(self.config.get("ngram_range", [1, 2]))
        
        # Stop words
        self.stop_words = set(self.config.get("stop_words", self._get_default_stop_words()))
        
        # Corpus storage for IDF calculation
        self.corpus = []
        self.idf_scores = {}
        self.vocabulary = set()
    
    def _get_default_stop_words(self) -> List[str]:
        """Get default English stop words."""
        return [
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with'
        ]
    
    def extract(self, documents: List[Document]) -> List[Document]:
        """Extract keywords using TF-IDF."""
        # Build corpus from all documents
        self._build_corpus(documents)
        
        # Calculate IDF scores
        self._calculate_idf()
        
        # Extract keywords for each document
        for doc in documents:
            try:
                keywords = self._extract_tfidf_keywords(doc.content)
                
                # Add to metadata
                if "extractors" not in doc.metadata:
                    doc.metadata["extractors"] = {}
                
                doc.metadata["extractors"]["tfidf_keywords"] = [
                    {"phrase": phrase, "score": score} 
                    for phrase, score in keywords
                ]
                doc.metadata["tfidf_keywords"] = [phrase for phrase, _ in keywords]
                
                self.logger.debug(f"Extracted {len(keywords)} TF-IDF keywords from document {doc.id}")
                
            except Exception as e:
                self.logger.error(f"TF-IDF extraction failed for document {doc.id}: {e}")
        
        return documents
    
    def _build_corpus(self, documents: List[Document]) -> None:
        """Build corpus from documents."""
        self.corpus = []
        for doc in documents:
            processed_text = self._preprocess_text(doc.content)
            self.corpus.append(processed_text)
            
            # Build vocabulary
            terms = self._extract_terms(processed_text)
            self.vocabulary.update(terms)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for TF-IDF."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _extract_terms(self, text: str) -> List[str]:
        """Extract terms (n-grams) from text."""
        words = text.split()
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        terms = []
        
        # Generate n-grams
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(words) - n + 1):
                term = ' '.join(words[i:i + n])
                terms.append(term)
        
        return terms
    
    def _calculate_idf(self) -> None:
        """Calculate IDF scores for all terms."""
        # Count document frequency for each term
        df = Counter()
        
        for doc_text in self.corpus:
            terms_in_doc = set(self._extract_terms(doc_text))
            for term in terms_in_doc:
                df[term] += 1
        
        # Calculate IDF
        num_docs = len(self.corpus)
        self.idf_scores = {}
        
        for term, doc_freq in df.items():
            # Filter by document frequency
            if doc_freq < self.min_df or doc_freq > self.max_df * num_docs:
                continue
            
            # Calculate IDF
            idf = math.log(num_docs / doc_freq)
            self.idf_scores[term] = idf
    
    def _extract_tfidf_keywords(self, text: str) -> List[tuple]:
        """Extract TF-IDF keywords from text."""
        processed_text = self._preprocess_text(text)
        terms = self._extract_terms(processed_text)
        
        # Calculate TF
        tf = Counter(terms)
        total_terms = len(terms)
        
        # Calculate TF-IDF scores
        tfidf_scores = []
        for term, freq in tf.items():
            if term in self.idf_scores:
                tf_score = freq / total_terms
                idf_score = self.idf_scores[term]
                tfidf_score = tf_score * idf_score
                tfidf_scores.append((term, tfidf_score))
        
        # Sort by score and return top terms
        tfidf_scores.sort(key=lambda x: x[1], reverse=True)
        
        return tfidf_scores[:self.max_features]
    
    def get_dependencies(self) -> List[str]:
        """TF-IDF has no external dependencies."""
        return []