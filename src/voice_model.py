import numpy as np
import pandas as pd
import librosa

from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


TARGET_SAMPLE_RATE = 16000
DEFAULT_THRESHOLD = 0.97165

MFCC_COLUMNS = [
    f"mfcc_{i}_{stat}"
    for i in range(1, 14)
    for stat in ("mean", "std")
]


class VoiceVerifier:
    """Verify a claimed speaker using enrolled MFCC voice profiles.

    Profiles can be loaded from a saved .npz archive. If no archive is
    supplied, they are built from original (non-augmented) rows in the
    feature CSV.
    """

    def __init__(self, feature_csv_path=None, threshold=None, profiles_path=None):
        self.voice_profiles = {}

        if profiles_path is not None:
            self._load_profiles(profiles_path)
            saved_threshold = self.threshold
            self.threshold = (
                float(threshold) if threshold is not None else saved_threshold
            )
        else:
            if feature_csv_path is None:
                raise ValueError("Provide profiles_path or feature_csv_path.")
            self.threshold = (
                float(threshold) if threshold is not None else DEFAULT_THRESHOLD
            )
            self._build_profiles(feature_csv_path)

        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1.")

    def _build_profiles(self, feature_csv_path):
        features_df = pd.read_csv(feature_csv_path)
        required_metadata = {"augmentation", "member_id"}
        missing_metadata = required_metadata - set(features_df.columns)
        if missing_metadata:
            raise ValueError(
                f"Missing required metadata columns: {sorted(missing_metadata)}"
            )

        original_df = features_df[
            features_df["augmentation"] == "original"
        ].copy()
        if original_df.empty:
            raise ValueError("No original recordings found.")

        missing_columns = [
            col for col in MFCC_COLUMNS if col not in original_df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing MFCC columns: {missing_columns}")

        for member_id, group in original_df.groupby("member_id"):
            vectors = group[MFCC_COLUMNS].to_numpy(dtype=float)
            self._validate_feature_matrix(vectors)
            normalized_vectors = normalize(vectors)
            average_profile = normalized_vectors.mean(axis=0)
            profile = normalize(average_profile.reshape(1, -1))[0]
            self.voice_profiles[str(member_id)] = profile

        if not self.voice_profiles:
            raise ValueError("No enrolled speaker profiles could be built.")

    def _load_profiles(self, profiles_path):
        profiles_path = Path(profiles_path)
        if not profiles_path.is_file():
            raise FileNotFoundError(
                f"Saved voice profiles not found: {profiles_path}"
            )

        try:
            with np.load(profiles_path, allow_pickle=False) as archive:
                required_keys = {"member_ids", "profiles", "threshold"}
                missing_keys = required_keys - set(archive.files)
                if missing_keys:
                    raise ValueError(
                        f"Profile archive is missing keys: {sorted(missing_keys)}"
                    )
                member_ids = archive["member_ids"].astype(str)
                profiles = np.asarray(archive["profiles"], dtype=float)
                saved_threshold = float(archive["threshold"])
        except (OSError, KeyError) as exc:
            raise ValueError(f"Could not read saved voice profiles: {exc}") from exc

        if profiles.ndim != 2 or profiles.shape[1] != len(MFCC_COLUMNS):
            raise ValueError(
                f"Expected profile shape (number_of_speakers, {len(MFCC_COLUMNS)}), "
                f"got {profiles.shape}."
            )
        if len(member_ids) != profiles.shape[0]:
            raise ValueError(
                "The number of member IDs does not match the number of profiles."
            )
        if len(set(member_ids.tolist())) != len(member_ids):
            raise ValueError("Duplicate member IDs found in profile archive.")
        self._validate_feature_matrix(profiles)
        if not 0 <= saved_threshold <= 1:
            raise ValueError("Saved threshold must be between 0 and 1.")

        normalized_profiles = normalize(profiles)
        self.voice_profiles = {
            member_id: profile
            for member_id, profile in zip(member_ids, normalized_profiles)
        }
        self.threshold = saved_threshold

    @staticmethod
    def _validate_feature_matrix(values):
        if values.size == 0:
            raise ValueError("Feature data is empty.")
        if not np.isfinite(values).all():
            raise ValueError("Feature data contains NaN or infinite values.")
        if np.any(np.linalg.norm(values, axis=1) == 0):
            raise ValueError("Feature data contains an all-zero vector.")

    def extract_features(self, audio_path):
        audio, sample_rate = librosa.load(
            audio_path,
            sr=TARGET_SAMPLE_RATE,
            mono=True,
        )
        audio, _ = librosa.effects.trim(audio, top_db=30)
        if audio.size == 0:
            raise ValueError(f"No usable audio signal found in: {audio_path}")

        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95

        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        feature_values = []
        for coefficient in mfcc:
            feature_values.append(np.mean(coefficient))
            feature_values.append(np.std(coefficient))

        features = np.asarray(feature_values, dtype=float)
        self._validate_feature_matrix(features.reshape(1, -1))
        return features

    def verify(self, audio_path, claimed_member_id):
        claimed_member_id = str(claimed_member_id)
        if claimed_member_id not in self.voice_profiles:
            return {
                "verified": False,
                "claimed_member_id": claimed_member_id,
                "similarity": None,
                "threshold": self.threshold,
                "message": "Speaker is not enrolled.",
            }

        query_features = self.extract_features(audio_path)
        query_vector = normalize(query_features.reshape(1, -1))
        speaker_profile = self.voice_profiles[claimed_member_id].reshape(1, -1)
        similarity = float(
            cosine_similarity(query_vector, speaker_profile)[0, 0]
        )
        verified = similarity >= self.threshold

        return {
            "verified": verified,
            "claimed_member_id": claimed_member_id,
            "similarity": similarity,
            "threshold": self.threshold,
            "message": (
                "Voice verified." if verified else "Voice verification failed."
            ),
        }
