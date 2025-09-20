"""Content statistics extraction for document analysis."""

import re
import string
import math
from collections import Counter
from typing import Dict, Any, List, Optional
import logging

from components.extractors.base import BaseExtractor
from core.base import Document

logger = logging.getLogger(__name__)


class ContentStatisticsExtractor(BaseExtractor):
    """
    Extract comprehensive content statistics from documents.

    Provides readability metrics, content analysis, and structural information
    without requiring external NLP models.
    """

    def __init__(
        self, name: str = "ContentStatistics", config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, config)

        # Configuration
        self.include_readability = self.config.get("include_readability", True)
        self.include_vocabulary = self.config.get("include_vocabulary", True)
        self.include_structure = self.config.get("include_structure", True)
        self.include_sentiment_indicators = self.config.get(
            "include_sentiment_indicators", True
        )
        self.language = self.config.get("language", "en")

        # Reading speed configuration (words per minute)
        self.reading_speeds = self.config.get(
            "reading_speeds", {"slow": 150, "average": 200, "fast": 300}
        )

        # Initialize word lists for sentiment analysis
        self.sentiment_words = self._initialize_sentiment_words()

        # Common English words for vocabulary analysis
        self.common_words = self._initialize_common_words()

    def _initialize_sentiment_words(self) -> Dict[str, List[str]]:
        """Initialize basic sentiment word lists."""
        return {
            "positive": [
                "good",
                "great",
                "excellent",
                "amazing",
                "wonderful",
                "fantastic",
                "awesome",
                "brilliant",
                "perfect",
                "outstanding",
                "superb",
                "happy",
                "pleased",
                "satisfied",
                "delighted",
                "thrilled",
                "excited",
                "love",
                "like",
                "enjoy",
                "appreciate",
                "thank",
                "thanks",
                "grateful",
                "success",
                "achieve",
                "win",
                "benefit",
                "improve",
                "better",
                "best",
            ],
            "negative": [
                "bad",
                "terrible",
                "awful",
                "horrible",
                "disgusting",
                "hate",
                "dislike",
                "angry",
                "frustrated",
                "annoyed",
                "disappointed",
                "sad",
                "unhappy",
                "upset",
                "worried",
                "concerned",
                "problem",
                "issue",
                "error",
                "mistake",
                "fail",
                "failure",
                "wrong",
                "broken",
                "damaged",
                "worst",
                "worse",
                "difficult",
                "hard",
                "impossible",
                "never",
                "unfortunately",
                "regret",
                "sorry",
                "apologize",
            ],
            "uncertainty": [
                "maybe",
                "perhaps",
                "possibly",
                "might",
                "could",
                "would",
                "should",
                "uncertain",
                "unclear",
                "confused",
                "doubt",
                "question",
                "wonder",
                "think",
                "believe",
                "assume",
                "guess",
                "suppose",
                "probably",
                "likely",
                "unlikely",
                "unsure",
                "hesitant",
            ],
            "urgency": [
                "urgent",
                "emergency",
                "critical",
                "immediate",
                "asap",
                "quickly",
                "fast",
                "hurry",
                "rush",
                "deadline",
                "time-sensitive",
                "priority",
                "important",
                "crucial",
                "vital",
                "essential",
                "must",
                "need",
                "required",
                "necessary",
                "now",
                "today",
                "tomorrow",
            ],
        }

    def _initialize_common_words(self) -> List[str]:
        """Initialize list of common English words."""
        return [
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "among",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
            "a",
            "an",
            "some",
            "any",
            "many",
            "much",
            "few",
            "little",
            "all",
            "every",
            "each",
            "one",
            "two",
            "three",
            "first",
            "second",
            "last",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "cannot",
            "must",
            "shall",
            "get",
            "got",
            "go",
            "went",
            "come",
            "came",
            "see",
            "saw",
            "know",
            "knew",
            "think",
            "thought",
            "say",
            "said",
            "tell",
            "told",
            "ask",
            "asked",
            "give",
            "gave",
            "take",
            "took",
            "make",
            "made",
            "put",
            "find",
            "found",
            "use",
            "used",
            "work",
            "worked",
            "call",
            "called",
            "try",
            "tried",
        ]

    def extract(self, documents: List[Document]) -> List[Document]:
        """Extract content statistics from documents."""
        for doc in documents:
            try:
                stats = self._extract_content_statistics(doc.content)

                # Add to metadata
                if "extractors" not in doc.metadata:
                    doc.metadata["extractors"] = {}

                doc.metadata["extractors"]["statistics"] = stats

                # Add top-level statistics for easy access
                doc.metadata["word_count"] = stats["basic"]["word_count"]
                doc.metadata["character_count"] = stats["basic"]["character_count"]
                doc.metadata["sentence_count"] = stats["basic"]["sentence_count"]
                doc.metadata["reading_time_minutes"] = stats["readability"][
                    "reading_time_minutes"
                ]

                if self.include_sentiment_indicators:
                    doc.metadata["sentiment_score"] = stats["sentiment"][
                        "overall_score"
                    ]

                self.logger.debug(f"Extracted content statistics for document {doc.id}")

            except Exception as e:
                self.logger.error(
                    f"Statistics extraction failed for document {doc.id}: {e}"
                )

        return documents

    def _extract_content_statistics(self, text: str) -> Dict[str, Any]:
        """Extract comprehensive content statistics."""
        stats = {}

        # Basic counts
        stats["basic"] = self._calculate_basic_stats(text)

        # Readability metrics
        if self.include_readability:
            stats["readability"] = self._calculate_readability_metrics(
                text, stats["basic"]
            )

        # Vocabulary analysis
        if self.include_vocabulary:
            stats["vocabulary"] = self._analyze_vocabulary(text)

        # Structural analysis
        if self.include_structure:
            stats["structure"] = self._analyze_structure(text)

        # Sentiment indicators
        if self.include_sentiment_indicators:
            stats["sentiment"] = self._analyze_sentiment_indicators(text)

        return stats

    def _calculate_basic_stats(self, text: str) -> Dict[str, Any]:
        """Calculate basic text statistics."""
        # Character counts
        char_count = len(text)
        char_count_no_spaces = len(text.replace(" ", ""))

        # Word analysis
        words = text.split()
        word_count = len(words)

        # Sentence analysis
        sentences = self._split_sentences(text)
        sentence_count = len([s for s in sentences if s.strip()])

        # Paragraph analysis
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        paragraph_count = len(paragraphs)

        # Line analysis
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        line_count = len(lines)

        # Average calculations
        avg_words_per_sentence = word_count / max(sentence_count, 1)
        avg_chars_per_word = char_count_no_spaces / max(word_count, 1)
        avg_sentences_per_paragraph = sentence_count / max(paragraph_count, 1)

        return {
            "character_count": char_count,
            "character_count_no_spaces": char_count_no_spaces,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "line_count": line_count,
            "avg_words_per_sentence": round(avg_words_per_sentence, 2),
            "avg_chars_per_word": round(avg_chars_per_word, 2),
            "avg_sentences_per_paragraph": round(avg_sentences_per_paragraph, 2),
            "whitespace_ratio": round(
                (char_count - char_count_no_spaces) / max(char_count, 1), 3
            ),
        }

    def _calculate_readability_metrics(
        self, text: str, basic_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate readability metrics."""
        word_count = basic_stats["word_count"]
        sentence_count = basic_stats["sentence_count"]

        # Syllable counting (approximation)
        syllable_count = self._count_syllables(text)

        # Flesch Reading Ease Score
        # Formula: 206.835 - (1.015 × ASL) - (84.6 × ASW)
        # ASL = Average Sentence Length, ASW = Average Syllables per Word
        if sentence_count > 0 and word_count > 0:
            asl = word_count / sentence_count
            asw = syllable_count / word_count
            flesch_score = 206.835 - (1.015 * asl) - (84.6 * asw)
            flesch_score = max(0, min(100, flesch_score))  # Clamp to 0-100
        else:
            flesch_score = 0

        # Flesch-Kincaid Grade Level
        # Formula: (0.39 × ASL) + (11.8 × ASW) - 15.59
        if sentence_count > 0 and word_count > 0:
            fk_grade = (0.39 * asl) + (11.8 * asw) - 15.59
            fk_grade = max(0, fk_grade)
        else:
            fk_grade = 0

        # Automated Readability Index
        # Formula: 4.71 × (characters/words) + 0.5 × (words/sentences) - 21.43
        if sentence_count > 0 and word_count > 0:
            ari = (4.71 * basic_stats["avg_chars_per_word"]) + (0.5 * asl) - 21.43
            ari = max(0, ari)
        else:
            ari = 0

        # Reading time estimates
        reading_times = {}
        for speed_name, wpm in self.reading_speeds.items():
            reading_times[f"reading_time_{speed_name}_minutes"] = round(
                word_count / wpm, 2
            )

        # Readability interpretation
        if flesch_score >= 90:
            reading_level = "Very Easy"
        elif flesch_score >= 80:
            reading_level = "Easy"
        elif flesch_score >= 70:
            reading_level = "Fairly Easy"
        elif flesch_score >= 60:
            reading_level = "Standard"
        elif flesch_score >= 50:
            reading_level = "Fairly Difficult"
        elif flesch_score >= 30:
            reading_level = "Difficult"
        else:
            reading_level = "Very Difficult"

        return {
            "flesch_reading_ease": round(flesch_score, 2),
            "flesch_kincaid_grade": round(fk_grade, 2),
            "automated_readability_index": round(ari, 2),
            "reading_level": reading_level,
            "syllable_count": syllable_count,
            "avg_syllables_per_word": round(syllable_count / max(word_count, 1), 2),
            **reading_times,
            "reading_time_minutes": reading_times.get(
                "reading_time_average_minutes", 0
            ),
        }

    def _analyze_vocabulary(self, text: str) -> Dict[str, Any]:
        """Analyze vocabulary complexity and diversity."""
        # Clean and tokenize
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

        if not words:
            return {"error": "No words found"}

        word_freq = Counter(words)
        unique_words = len(word_freq)
        total_words = len(words)

        # Type-Token Ratio (vocabulary diversity)
        ttr = unique_words / total_words if total_words > 0 else 0

        # Most common words
        most_common = word_freq.most_common(10)

        # Hapax legomena (words that appear only once)
        hapax_legomena = sum(1 for count in word_freq.values() if count == 1)
        hapax_ratio = hapax_legomena / unique_words if unique_words > 0 else 0

        # Complex words (words with 3+ syllables)
        complex_words = [
            word for word in words if self._count_word_syllables(word) >= 3
        ]
        complex_word_ratio = len(complex_words) / total_words if total_words > 0 else 0

        # Common vs uncommon words
        common_word_count = sum(1 for word in words if word in self.common_words)
        common_word_ratio = common_word_count / total_words if total_words > 0 else 0

        # Average word length
        avg_word_length = (
            sum(len(word) for word in words) / total_words if total_words > 0 else 0
        )

        return {
            "unique_words": unique_words,
            "total_words": total_words,
            "type_token_ratio": round(ttr, 3),
            "hapax_legomena": hapax_legomena,
            "hapax_ratio": round(hapax_ratio, 3),
            "complex_words": len(complex_words),
            "complex_word_ratio": round(complex_word_ratio, 3),
            "common_word_ratio": round(common_word_ratio, 3),
            "avg_word_length": round(avg_word_length, 2),
            "most_common_words": most_common,
            "vocabulary_richness": "high"
            if ttr > 0.5
            else "medium"
            if ttr > 0.3
            else "low",
        }

    def _analyze_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure."""
        # Punctuation analysis
        punctuation_counts = {}
        for punct in string.punctuation:
            count = text.count(punct)
            if count > 0:
                punctuation_counts[punct] = count

        # Question and exclamation analysis
        questions = text.count("?")
        exclamations = text.count("!")
        periods = text.count(".")

        # Capitalization analysis
        uppercase_words = len(re.findall(r"\b[A-Z]{2,}\b", text))
        title_case_words = len(re.findall(r"\b[A-Z][a-z]+\b", text))

        # Number analysis
        numbers = re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", text)
        number_count = len(numbers)

        # URL and email detection
        urls = len(re.findall(r"http[s]?://\S+", text))
        emails = len(
            re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
        )

        # List detection (simple heuristic)
        bullet_points = len(re.findall(r"^\s*[•\-\*]\s+", text, re.MULTILINE))
        numbered_lists = len(re.findall(r"^\s*\d+\.?\s+", text, re.MULTILINE))

        # White space patterns
        double_spaces = text.count("  ")
        tabs = text.count("\t")

        return {
            "punctuation_counts": punctuation_counts,
            "questions": questions,
            "exclamations": exclamations,
            "periods": periods,
            "uppercase_words": uppercase_words,
            "title_case_words": title_case_words,
            "numbers": number_count,
            "urls": urls,
            "emails": emails,
            "bullet_points": bullet_points,
            "numbered_lists": numbered_lists,
            "double_spaces": double_spaces,
            "tabs": tabs,
            "formatting_indicators": {
                "has_lists": bullet_points > 0 or numbered_lists > 0,
                "has_questions": questions > 0,
                "has_emphasis": exclamations > 0 or uppercase_words > 0,
                "has_structure": bullet_points + numbered_lists + questions > 0,
            },
        }

    def _analyze_sentiment_indicators(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment indicators in text."""
        text_lower = text.lower()
        words = re.findall(r"\b[a-zA-Z]+\b", text_lower)

        sentiment_counts = {}
        sentiment_words_found = {}

        for sentiment_type, word_list in self.sentiment_words.items():
            count = sum(1 for word in words if word in word_list)
            sentiment_counts[sentiment_type] = count
            sentiment_words_found[sentiment_type] = [
                word for word in words if word in word_list
            ]

        total_words = len(words)
        sentiment_ratios = {}

        for sentiment_type, count in sentiment_counts.items():
            sentiment_ratios[f"{sentiment_type}_ratio"] = (
                count / total_words if total_words > 0 else 0
            )

        # Overall sentiment score (positive - negative)
        positive_score = sentiment_ratios.get("positive_ratio", 0)
        negative_score = sentiment_ratios.get("negative_ratio", 0)
        overall_score = positive_score - negative_score

        # Sentiment classification
        if overall_score > 0.02:
            sentiment_classification = "positive"
        elif overall_score < -0.02:
            sentiment_classification = "negative"
        else:
            sentiment_classification = "neutral"

        return {
            "sentiment_counts": sentiment_counts,
            "sentiment_ratios": sentiment_ratios,
            "sentiment_words_found": sentiment_words_found,
            "overall_score": round(overall_score, 4),
            "sentiment_classification": sentiment_classification,
            "urgency_indicators": sentiment_counts.get("urgency", 0),
            "uncertainty_indicators": sentiment_counts.get("uncertainty", 0),
        }

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _count_syllables(self, text: str) -> int:
        """Count syllables in text (approximation)."""
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        total_syllables = sum(self._count_word_syllables(word) for word in words)
        return total_syllables

    def _count_word_syllables(self, word: str) -> int:
        """Count syllables in a word (approximation)."""
        word = word.lower()
        count = 0
        vowels = "aeiouy"

        if word[0] in vowels:
            count += 1

        for i in range(1, len(word)):
            if word[i] in vowels and word[i - 1] not in vowels:
                count += 1

        if word.endswith("e"):
            count -= 1

        if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
            count += 1

        if count == 0:
            count += 1

        return count

    def get_dependencies(self) -> List[str]:
        """No external dependencies required."""
        return []
