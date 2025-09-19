"""
Summary Extractor

Extracts summaries and key information from text content without using LLMs.
Uses statistical and rule-based approaches for summarization.
"""

import re
from typing import Dict, Any, List, Optional
from collections import Counter
import math

from rag.components.extractors.base import BaseExtractor


class SummaryExtractor(BaseExtractor):
    """Extractor for generating document summaries using statistical methods."""
    
    def __init__(self, name: str = "SummaryExtractor", config: Optional[Dict[str, Any]] = None):
        """
        Initialize summary extractor.
        
        Args:
            name: Extractor name
            config: Extractor configuration
        """
        super().__init__(name=name, config=config)
        self.summary_sentences = config.get("summary_sentences", 3) if config else 3
        self.min_sentence_length = config.get("min_sentence_length", 10) if config else 10
        self.max_sentence_length = config.get("max_sentence_length", 500) if config else 500
        self.include_key_phrases = config.get("include_key_phrases", True) if config else True
        self.include_statistics = config.get("include_statistics", True) if config else True
        
        # Common stop words for filtering
        self.stop_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'this', 'that', 'these', 'those', 'i', 'me',
            'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is',
            'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
            'do', 'does', 'did', 'doing', 'will', 'would', 'should', 'could', 'can', 'may',
            'might', 'must', 'shall'
        ])
    
    def extract(self, documents: List['Document']) -> List['Document']:
        """Extract summaries from documents."""
        from rag.core.base import Document
        
        for doc in documents:
            try:
                summary_data = self._extract_from_text(doc.content, doc.metadata)
                
                # Add to extractors metadata
                if "extractors" not in doc.metadata:
                    doc.metadata["extractors"] = {}
                
                doc.metadata["extractors"]["summary"] = summary_data
                
                self.logger.debug(f"Extracted summary for document {doc.id}")
                
            except Exception as e:
                self.logger.error(f"Summary extraction failed for document {doc.id}: {e}")
        
        return documents
    
    def _extract_from_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract summary information from text.
        
        Args:
            text: Input text to summarize
            metadata: Optional metadata dictionary
            
        Returns:
            Dictionary containing summary information
        """
        if not text or not text.strip():
            return {}
        
        try:
            # Split into sentences
            sentences = self._split_sentences(text)
            if not sentences:
                return {}
            
            # Calculate sentence scores
            sentence_scores = self._calculate_sentence_scores(sentences)
            
            # Select top sentences for summary
            summary_sentences = self._select_summary_sentences(sentences, sentence_scores)
            
            result = {
                "extractive_summary": " ".join(summary_sentences),
                "sentence_count": len(sentences),
                "avg_sentence_length": sum(len(s.split()) for s in sentences) / len(sentences),
                "summary_ratio": len(summary_sentences) / len(sentences)
            }
            
            # Add key phrases if requested
            if self.include_key_phrases:
                key_phrases = self._extract_key_phrases(text)
                result["key_phrases"] = key_phrases
            
            # Add statistics if requested
            if self.include_statistics:
                stats = self._calculate_text_statistics(text, sentences)
                result.update(stats)
            
            # Add first and last sentences as context
            if len(sentences) > 1:
                result["opening_sentence"] = sentences[0]
                result["closing_sentence"] = sentences[-1]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error extracting summary: {e}")
            return {}
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Basic sentence splitting using regex
        # This is a simplified approach - for production, consider using spaCy or NLTK
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) >= self.min_sentence_length and 
                len(sentence) <= self.max_sentence_length and
                not sentence.isdigit()):
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _calculate_sentence_scores(self, sentences: List[str]) -> Dict[int, float]:
        """Calculate importance scores for sentences using TF-IDF-like approach."""
        if not sentences:
            return {}
        
        # Tokenize all sentences
        word_freq = Counter()
        sentence_words = []
        
        for sentence in sentences:
            words = self._tokenize(sentence)
            sentence_words.append(words)
            word_freq.update(words)
        
        # Calculate IDF-like scores (rarer words are more important)
        total_sentences = len(sentences)
        word_importance = {}
        
        for word, freq in word_freq.items():
            # Simple importance score: log(total_sentences / word_frequency)
            if freq > 0:
                word_importance[word] = math.log(total_sentences / freq)
            else:
                word_importance[word] = 0
        
        # Score sentences based on word importance
        sentence_scores = {}
        for i, words in enumerate(sentence_words):
            if not words:
                sentence_scores[i] = 0
                continue
                
            # Sum of word importance scores, normalized by sentence length
            score = sum(word_importance.get(word, 0) for word in words)
            sentence_scores[i] = score / len(words) if words else 0
            
            # Boost score for sentences with numbers (often contain facts)
            if re.search(r'\d+', sentences[i]):
                sentence_scores[i] *= 1.2
            
            # Boost score for sentences in first 20% or last 20% of text
            position_factor = 1.0
            if i < len(sentences) * 0.2:  # First 20%
                position_factor = 1.3
            elif i > len(sentences) * 0.8:  # Last 20%
                position_factor = 1.1
            
            sentence_scores[i] *= position_factor
        
        return sentence_scores
    
    def _select_summary_sentences(self, sentences: List[str], scores: Dict[int, float]) -> List[str]:
        """Select top sentences for summary based on scores."""
        if not sentences or not scores:
            return []
        
        # Sort sentences by score (descending)
        sorted_sentences = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top sentences, but maintain original order
        selected_indices = sorted([idx for idx, score in sorted_sentences[:self.summary_sentences]])
        
        return [sentences[i] for i in selected_indices if i < len(sentences)]
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases using simple n-gram analysis."""
        words = self._tokenize(text)
        if len(words) < 2:
            return []
        
        # Extract bigrams and trigrams
        phrases = []
        
        # Bigrams
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            if not any(word in self.stop_words for word in [words[i], words[i+1]]):
                phrases.append(phrase)
        
        # Trigrams
        for i in range(len(words) - 2):
            phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
            if not any(word in self.stop_words for word in [words[i], words[i+1], words[i+2]]):
                phrases.append(phrase)
        
        # Count phrase frequency and return most common
        phrase_counts = Counter(phrases)
        return [phrase for phrase, count in phrase_counts.most_common(10) if count > 1]
    
    def _calculate_text_statistics(self, text: str, sentences: List[str]) -> Dict[str, Any]:
        """Calculate various text statistics."""
        words = self._tokenize(text)
        
        if not words:
            return {}
        
        # Basic statistics
        stats = {
            "word_count": len(words),
            "unique_words": len(set(words)),
            "avg_word_length": sum(len(word) for word in words) / len(words),
            "lexical_diversity": len(set(words)) / len(words) if words else 0,
            "longest_sentence": max(len(s.split()) for s in sentences) if sentences else 0,
            "shortest_sentence": min(len(s.split()) for s in sentences) if sentences else 0
        }
        
        # Character statistics
        stats.update({
            "character_count": len(text),
            "character_count_no_spaces": len(text.replace(' ', '')),
            "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
        })
        
        # Reading time estimation (assuming 200 words per minute)
        stats["estimated_reading_time_minutes"] = len(words) / 200
        
        # Complexity indicators
        long_words = [word for word in words if len(word) > 6]
        stats["long_words_ratio"] = len(long_words) / len(words) if words else 0
        
        # Sentence complexity
        if sentences:
            sentence_lengths = [len(s.split()) for s in sentences]
            stats["avg_sentence_complexity"] = sum(sentence_lengths) / len(sentence_lengths)
        
        return stats
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization - split on whitespace and punctuation."""
        # Convert to lowercase and split on whitespace/punctuation
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter out stop words and short words
        return [token for token in tokens if token not in self.stop_words and len(token) > 2]
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names this extractor produces."""
        features = [
            "extractive_summary",
            "sentence_count", 
            "avg_sentence_length",
            "summary_ratio",
            "opening_sentence",
            "closing_sentence"
        ]
        
        if self.include_key_phrases:
            features.append("key_phrases")
        
        if self.include_statistics:
            features.extend([
                "word_count",
                "unique_words", 
                "avg_word_length",
                "lexical_diversity",
                "longest_sentence",
                "shortest_sentence",
                "character_count",
                "character_count_no_spaces",
                "paragraph_count",
                "estimated_reading_time_minutes",
                "long_words_ratio",
                "avg_sentence_complexity"
            ])
        
        return features
    
    @staticmethod
    def get_config_schema() -> Dict[str, Any]:
        """Get configuration schema for the extractor."""
        return {
            "type": "object",
            "properties": {
                "summary_sentences": {
                    "type": "integer",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Number of sentences to include in summary"
                },
                "min_sentence_length": {
                    "type": "integer", 
                    "default": 10,
                    "description": "Minimum sentence length to consider"
                },
                "max_sentence_length": {
                    "type": "integer",
                    "default": 500,
                    "description": "Maximum sentence length to consider"
                },
                "include_key_phrases": {
                    "type": "boolean",
                    "default": True,
                    "description": "Extract key phrases from text"
                },
                "include_statistics": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include text statistics"
                }
            }
        }
    
    def get_dependencies(self) -> List[str]:
        """Get list of required dependencies."""
        return []  # No external dependencies required