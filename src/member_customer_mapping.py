
"""
Maps verified team member identities to existing customer IDs.

These assignments are for the academic system demonstration only.
They do not represent verified real-world customer identities.
"""

MEMBER_CUSTOMER_MAPPING = {
    "member_1": 161,
    "member_2": 159,
    "member_3": 187,
}


def get_customer_id(member_id: str) -> int:
    """Return the assigned customer ID for a verified member."""
    if member_id not in MEMBER_CUSTOMER_MAPPING:
        raise ValueError(f"No customer mapping found for {member_id}")

    return MEMBER_CUSTOMER_MAPPING[member_id]