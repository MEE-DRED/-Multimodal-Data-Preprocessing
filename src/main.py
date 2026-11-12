
from pathlib import Path
import argparse
import json

from face_model import verify_face
from voice_model import VoiceVerifier
from member_customer_mapping import get_customer_id
from product_model import predict_product


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOICE_PROFILES_PATH = PROJECT_ROOT / "models" / "voice_profiles.npz"


def run_multimodal_system(image_path, audio_path):
    """
    Run face verification, voice verification, and product prediction.

    Product prediction is allowed only when both identity checks pass.
    """

    # STEP 1: Verify the face
    face_result = verify_face(image_path)

    if not face_result.get("verified"):
        return {
            "success": False,
            "stage": "face_verification",
            "message": "Face verification failed. Transaction rejected.",
            "face_result": face_result,
        }

    member_id = face_result.get("user_id")

    if not member_id:
        return {
            "success": False,
            "stage": "face_verification",
            "message": "No valid member identity was returned.",
        }

    # STEP 2: Verify the voice against the face-recognized member
    try:
        voice_verifier = VoiceVerifier(
            profiles_path=VOICE_PROFILES_PATH
        )
        voice_result = voice_verifier.verify(
            audio_path,
            member_id
        )
    except Exception as exc:
        return {
            "success": False,
            "stage": "voice_verification",
            "message": (
                f"Voice verification could not run: "
                f"{type(exc).__name__}"
            ),
        }

    if not voice_result.get("verified"):
        return {
            "success": False,
            "stage": "voice_verification",
            "message": "Voice verification failed. Transaction rejected.",
            "face_result": face_result,
            "voice_result": voice_result,
        }

    # STEP 3: Confirm both components refer to the same member
    if voice_result.get("claimed_member_id") != member_id:
        return {
            "success": False,
            "stage": "identity_matching",
            "message": "Face and voice identities do not match.",
            "face_result": face_result,
            "voice_result": voice_result,
        }

    # STEP 4: Map the verified member to a customer ID
    try:
        customer_id = get_customer_id(member_id)
    except ValueError as exc:
        return {
            "success": False,
            "stage": "customer_mapping",
            "message": str(exc),
        }

    # STEP 5: Predict the product
    try:
        product_result = predict_product(customer_id)
    except Exception as exc:
        return {
            "success": False,
            "stage": "product_prediction",
            "message": f"Product prediction failed: {type(exc).__name__}",
        }

    return {
        "success": True,
        "stage": "completed",
        "message": "Face and voice verified. Product prediction completed.",
        "member_id": member_id,
        "customer_id": customer_id,
        "face_result": face_result,
        "voice_result": voice_result,
        "product_result": product_result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multimodal face and voice verification system"
    )
    parser.add_argument("--image", required=True, help="Path to face image")
    parser.add_argument("--audio", required=True, help="Path to voice recording")

    args = parser.parse_args()

    result = run_multimodal_system(args.image, args.audio)

    print(json.dumps(result, indent=4, default=str))