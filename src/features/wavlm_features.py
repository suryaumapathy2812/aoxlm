"""
WavLM-Based Pronunciation Feature Extraction

Uses WavLM embeddings for pronunciation quality assessment.
WavLM captures phonetic information better than confidence scores alone.

Key Features:
1. Frame-level embedding consistency - stable pronunciation
2. Embedding variance - pronunciation clarity
3. Temporal smoothness - natural speech flow
4. Layer-wise features - different layers capture different aspects

Usage:
    >>> from src.features.wavlm_features import WavLMFeatureExtractor
    >>> extractor = WavLMFeatureExtractor(device="cuda")
    >>> features = extractor.extract("audio.mp3")
    >>> print(features.embedding_consistency)  # 0.85

References:
- Chen et al. (2022): "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing"
- Microsoft WavLM: https://github.com/microsoft/unilm/tree/master/wavlm
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class WavLMFeatures:
    """Features extracted from WavLM embeddings."""

    # Embedding quality metrics
    embedding_mean_norm: float = 0.0  # Average embedding magnitude
    embedding_std: float = 0.0  # Embedding variance (consistency)
    embedding_consistency: float = 0.0  # Frame-to-frame similarity (0-1)

    # Temporal smoothness
    temporal_smoothness: float = 0.0  # How smooth transitions are (0-1)
    frame_variance: float = 0.0  # Variance across frames

    # Pronunciation quality proxies
    low_norm_ratio: float = 0.0  # % of frames with low embedding norm (unclear)
    high_variance_ratio: float = 0.0  # % of high-variance frames (unstable)

    # Statistics
    num_frames: int = 0
    duration: float = 0.0
    frame_rate: float = 50.0  # WavLM outputs ~50 frames/sec

    # Layer-wise features (optional)
    layer_norms: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class WavLMPronunciationScore:
    """Pronunciation assessment from WavLM."""

    score: float  # 0-100
    level: str  # CEFR level
    confidence: float  # 0-1
    features: WavLMFeatures = field(default_factory=WavLMFeatures)
    sub_scores: Dict[str, float] = field(default_factory=dict)
    feedback: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "score": round(self.score, 1),
            "level": self.level,
            "confidence": round(self.confidence, 3),
            "features": self.features.to_dict(),
            "sub_scores": {k: round(v, 1) for k, v in self.sub_scores.items()},
            "feedback": self.feedback,
        }


class WavLMFeatureExtractor:
    """
    Extract pronunciation features from WavLM embeddings.

    WavLM embeddings capture phonetic information that can indicate:
    - Pronunciation clarity (embedding magnitude/consistency)
    - Speech naturalness (temporal smoothness)
    - Articulation quality (frame-level patterns)

    Example:
        >>> extractor = WavLMFeatureExtractor(device="cuda")
        >>> features = extractor.extract("audio.mp3")
        >>> print(f"Consistency: {features.embedding_consistency:.2f}")
    """

    def __init__(
        self,
        model_name: str = "wavlm-large",
        device: Optional[str] = None,
        sample_rate: int = 16000,
    ):
        """
        Initialize WavLM feature extractor.

        Args:
            model_name: WavLM model variant (wavlm-large, wavlm-base)
            device: Device to use (cuda/cpu)
            sample_rate: Audio sample rate
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = sample_rate
        self.encoder = None  # Lazy loading
        self._loaded = False

    def _load_model(self):
        """Lazy load WavLM model."""
        if self._loaded:
            return

        from src.models.encoder import SemanticEncoder

        print(f"Loading WavLM ({self.model_name})...")
        self.encoder = SemanticEncoder(
            encoder_name=self.model_name,
            freeze=True,
            device=self.device,
        )
        self._loaded = True
        print(f"WavLM loaded on {self.device}")

    def extract(
        self,
        audio_path: Optional[str] = None,
        audio_array: Optional[np.ndarray] = None,
        return_embeddings: bool = False,
    ) -> Tuple[WavLMFeatures, Optional[np.ndarray]]:
        """
        Extract WavLM features from audio.

        Args:
            audio_path: Path to audio file
            audio_array: Pre-loaded audio array (16kHz)
            return_embeddings: Whether to return raw embeddings

        Returns:
            Tuple of (WavLMFeatures, optional embeddings)
        """
        self._load_model()

        # Load audio
        if audio_array is None:
            import librosa

            audio_array, _ = librosa.load(audio_path, sr=self.sample_rate)

        duration = len(audio_array) / self.sample_rate

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_array).float()

        # Get embeddings [1, T, D]
        with torch.no_grad():
            embeddings = self.encoder(audio_tensor, sampling_rate=self.sample_rate)

        embeddings = embeddings.squeeze(0)  # [T, D]
        embeddings_np = embeddings.cpu().numpy()

        # Extract features
        features = self._compute_features(embeddings_np, duration)

        if return_embeddings:
            return features, embeddings_np
        return features, None

    def _compute_features(
        self,
        embeddings: np.ndarray,
        duration: float,
    ) -> WavLMFeatures:
        """
        Compute pronunciation features from embeddings.

        Args:
            embeddings: Frame-level embeddings [T, D]
            duration: Audio duration in seconds

        Returns:
            WavLMFeatures
        """
        num_frames, dim = embeddings.shape

        # 1. EMBEDDING MAGNITUDE
        # Higher/more consistent magnitude often indicates clearer speech
        frame_norms = np.linalg.norm(embeddings, axis=1)
        mean_norm = float(np.mean(frame_norms))
        std_norm = float(np.std(frame_norms))

        # Normalize for consistency score
        norm_cv = std_norm / mean_norm if mean_norm > 0 else 0
        # Lower CV = more consistent = better
        embedding_consistency = max(0, 1 - norm_cv)

        # 2. FRAME VARIANCE
        # Overall variance in embeddings
        frame_variance = float(np.mean(np.var(embeddings, axis=1)))

        # 3. TEMPORAL SMOOTHNESS
        # Cosine similarity between adjacent frames
        if num_frames > 1:
            similarities = []
            for i in range(num_frames - 1):
                sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
                similarities.append(sim)
            temporal_smoothness = float(np.mean(similarities))
        else:
            temporal_smoothness = 0.5

        # 4. LOW NORM RATIO
        # Frames with unusually low norm might indicate unclear pronunciation
        norm_threshold = mean_norm * 0.7
        low_norm_count = np.sum(frame_norms < norm_threshold)
        low_norm_ratio = low_norm_count / num_frames if num_frames > 0 else 0

        # 5. HIGH VARIANCE RATIO
        # Frames with high variance might indicate unstable pronunciation
        frame_variances = np.var(embeddings, axis=1)
        var_threshold = np.percentile(frame_variances, 75)
        high_var_count = np.sum(frame_variances > var_threshold * 1.5)
        high_variance_ratio = high_var_count / num_frames if num_frames > 0 else 0

        return WavLMFeatures(
            embedding_mean_norm=mean_norm,
            embedding_std=std_norm,
            embedding_consistency=float(embedding_consistency),
            temporal_smoothness=float(temporal_smoothness),
            frame_variance=frame_variance,
            low_norm_ratio=float(low_norm_ratio),
            high_variance_ratio=float(high_variance_ratio),
            num_frames=num_frames,
            duration=duration,
            frame_rate=num_frames / duration if duration > 0 else 50.0,
        )

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a > 0 and norm_b > 0:
            return float(dot / (norm_a * norm_b))
        return 0.0


class WavLMPronunciationScorer:
    """
    Score pronunciation quality from WavLM features.

    Scoring dimensions:
    1. Embedding Consistency (30%) - stable pronunciation
    2. Temporal Smoothness (30%) - natural speech flow
    3. Articulation Quality (40%) - clear articulation

    CEFR Mapping:
    - A1-A2: Low consistency, choppy transitions
    - B1: Moderate consistency, developing smoothness
    - B2+: High consistency, smooth natural flow
    """

    def __init__(self):
        self.weights = {
            "consistency": 0.30,
            "smoothness": 0.30,
            "articulation": 0.40,
        }

    def score(self, features: WavLMFeatures) -> WavLMPronunciationScore:
        """
        Score pronunciation from WavLM features.

        Args:
            features: Extracted WavLM features

        Returns:
            WavLMPronunciationScore
        """
        if features.num_frames == 0:
            return WavLMPronunciationScore(
                score=50.0,
                level="A2",
                confidence=0.0,
                features=features,
            )

        # Calculate sub-scores
        sub_scores = {
            "consistency": self._score_consistency(features),
            "smoothness": self._score_smoothness(features),
            "articulation": self._score_articulation(features),
        }

        # Weighted average
        total_score = sum(sub_scores[k] * self.weights[k] for k in self.weights.keys())

        # Map to CEFR
        level = self._score_to_level(total_score)

        # Confidence based on duration
        confidence = min(0.95, 0.5 + features.duration / 60)

        # Generate feedback
        feedback = self._generate_feedback(features, sub_scores)

        return WavLMPronunciationScore(
            score=total_score,
            level=level,
            confidence=confidence,
            features=features,
            sub_scores=sub_scores,
            feedback=feedback,
        )

    def _score_consistency(self, features: WavLMFeatures) -> float:
        """Score embedding consistency (0-100)."""
        consistency = features.embedding_consistency

        # Map 0-1 to 0-100 with appropriate scaling
        if consistency >= 0.95:
            return 95
        elif consistency >= 0.90:
            return 80 + (consistency - 0.90) * 300  # 80-95
        elif consistency >= 0.80:
            return 60 + (consistency - 0.80) * 200  # 60-80
        elif consistency >= 0.70:
            return 45 + (consistency - 0.70) * 150  # 45-60
        else:
            return max(20, consistency * 65)  # 0-45

    def _score_smoothness(self, features: WavLMFeatures) -> float:
        """Score temporal smoothness (0-100)."""
        smoothness = features.temporal_smoothness

        # Cosine similarity typically 0.7-0.99 for speech
        if smoothness >= 0.95:
            return 95
        elif smoothness >= 0.90:
            return 80 + (smoothness - 0.90) * 300  # 80-95
        elif smoothness >= 0.85:
            return 65 + (smoothness - 0.85) * 300  # 65-80
        elif smoothness >= 0.75:
            return 45 + (smoothness - 0.75) * 200  # 45-65
        else:
            return max(20, smoothness * 60)  # 0-45

    def _score_articulation(self, features: WavLMFeatures) -> float:
        """Score articulation quality (0-100)."""
        # Lower ratios = better articulation
        low_norm_penalty = features.low_norm_ratio * 50
        high_var_penalty = features.high_variance_ratio * 30

        base_score = 85
        score = base_score - low_norm_penalty - high_var_penalty

        return max(20, min(100, score))

    def _score_to_level(self, score: float) -> str:
        """Convert score to CEFR level."""
        if score >= 75:
            return "B2+"
        elif score >= 55:
            return "B1"
        elif score >= 35:
            return "A2"
        else:
            return "A1"

    def _generate_feedback(
        self,
        features: WavLMFeatures,
        sub_scores: Dict[str, float],
    ) -> List[str]:
        """Generate feedback based on WavLM features."""
        feedback = []

        # Consistency feedback
        if sub_scores["consistency"] < 50:
            feedback.append(
                "Pronunciation consistency needs improvement. "
                "Practice speaking with steady, clear articulation."
            )
        elif sub_scores["consistency"] >= 80:
            feedback.append("Good pronunciation consistency - stable articulation.")

        # Smoothness feedback
        if sub_scores["smoothness"] < 50:
            feedback.append(
                "Speech transitions are choppy. "
                "Practice connecting words more smoothly."
            )
        elif sub_scores["smoothness"] >= 80:
            feedback.append("Good speech flow - smooth transitions between sounds.")

        # Articulation feedback
        if sub_scores["articulation"] < 50:
            feedback.append(
                "Some unclear articulation detected. "
                "Focus on pronouncing each word clearly."
            )
        elif sub_scores["articulation"] >= 80:
            feedback.append("Clear articulation - words are well pronounced.")

        return feedback


class WavLMPronunciationAssessor:
    """
    Complete WavLM-based pronunciation assessment.

    Combines feature extraction and scoring.

    Example:
        >>> assessor = WavLMPronunciationAssessor(device="cuda")
        >>> result = assessor.assess("audio.mp3")
        >>> print(f"Level: {result.level}, Score: {result.score:.1f}")
    """

    def __init__(
        self,
        model_name: str = "wavlm-large",
        device: Optional[str] = None,
    ):
        self.extractor = WavLMFeatureExtractor(
            model_name=model_name,
            device=device,
        )
        self.scorer = WavLMPronunciationScorer()

    def assess(
        self,
        audio_path: Optional[str] = None,
        audio_array: Optional[np.ndarray] = None,
    ) -> WavLMPronunciationScore:
        """
        Assess pronunciation from audio.

        Args:
            audio_path: Path to audio file
            audio_array: Pre-loaded audio array

        Returns:
            WavLMPronunciationScore
        """
        features, _ = self.extractor.extract(
            audio_path=audio_path,
            audio_array=audio_array,
        )
        return self.scorer.score(features)


# Convenience function
def assess_pronunciation_wavlm(
    audio_path: str,
    device: Optional[str] = None,
) -> WavLMPronunciationScore:
    """
    Quick pronunciation assessment using WavLM.

    Args:
        audio_path: Path to audio file
        device: Device to use

    Returns:
        WavLMPronunciationScore
    """
    assessor = WavLMPronunciationAssessor(device=device)
    return assessor.assess(audio_path=audio_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python wavlm_features.py <audio_path> [device]")
        sys.exit(1)

    audio_path = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else None

    print("=" * 60)
    print("WavLM Pronunciation Assessment")
    print("=" * 60)
    print(f"Audio: {audio_path}")
    print(f"Device: {device or 'auto'}")
    print()

    assessor = WavLMPronunciationAssessor(device=device)
    result = assessor.assess(audio_path=audio_path)

    print(f"Score: {result.score:.1f}/100")
    print(f"Level: {result.level}")
    print(f"Confidence: {result.confidence:.1%}")
    print()
    print("Sub-scores:")
    for name, score in result.sub_scores.items():
        print(f"  {name:15}: {score:.1f}")
    print()
    print("Features:")
    print(f"  Embedding consistency: {result.features.embedding_consistency:.3f}")
    print(f"  Temporal smoothness:   {result.features.temporal_smoothness:.3f}")
    print(f"  Low norm ratio:        {result.features.low_norm_ratio:.3f}")
    print(f"  Frames:                {result.features.num_frames}")
    print(f"  Duration:              {result.features.duration:.1f}s")
    print()
    if result.feedback:
        print("Feedback:")
        for fb in result.feedback:
            print(f"  - {fb}")
