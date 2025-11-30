import re

# Pre-compile the regex
CHARS_NUMBERS_PATTERN = re.compile(r"^[A-Za-z0-9_\-\.]+$")
CHARS_NUMBERS_WITH_SPACES_PATTERN = re.compile(r"^[A-Za-z0-9_\-\. ]+$")

ONLY_CHARS_PATTERN = re.compile(r"^[A-Za-z]+$")
ONLY_CHARS_WITH_SPACES_PATTERN = re.compile(r"^[A-Za-z ]+$")

ONLY_NUMBERS_PATTERN = re.compile(r"^\d+$")
ONLY_NUMBERS_WITH_SPACES_PATTERN = re.compile(r"^[\d ]+$")


MODE_CHARS_NUMS = "chars_and_numbers"
MODE_ONLY_CHARS = "only_chars"
MODE_ONLY_NUMS = "only_numbers"


def check_input_validation(
    text: str, 
    mode: str, 
    allow_spaces: bool = False
) -> tuple[bool, str]:
    """
    Validates input text based on the selected mode, length, and space allowance.

    Args:
        text: The string to validate.
        mode: Validation type — one of:
              'chars_and_numbers', 'only_chars', or 'only_numbers'.
        allow_spaces: Whether spaces are allowed in the input (default: False).

    Returns:
        (is_valid, error_message):
            - is_valid (bool): True if input passes all checks, else False.
            - error_message (str): Explanation if invalid, or empty string if valid.
    """
    # Basic checks
    if not text or text.strip() == "":
        return False, "Input cannot be empty."
    if len(text.strip()) < 4:
        return False, "Input must be at least 4 characters long."

    # Select appropriate regex and allowed characters description
    if mode == MODE_CHARS_NUMS:
        pattern = CHARS_NUMBERS_WITH_SPACES_PATTERN if allow_spaces else CHARS_NUMBERS_PATTERN
        allowed = "letters, numbers, underscores (_), hyphens (-), and dots (.)"
    elif mode == MODE_ONLY_CHARS:
        pattern = ONLY_CHARS_WITH_SPACES_PATTERN if allow_spaces else ONLY_CHARS_PATTERN
        allowed = "letters only (A–Z or a–z)"
    elif mode == MODE_ONLY_NUMS:
        pattern = ONLY_NUMBERS_WITH_SPACES_PATTERN if allow_spaces else ONLY_NUMBERS_PATTERN
        allowed = "numbers only (0–9)"
    else:
        raise ValueError(
            f"Invalid mode: '{mode}'. Must be one of: "
            f"'{MODE_CHARS_NUMS}', '{MODE_ONLY_CHARS}', or '{MODE_ONLY_NUMS}'."
        )

    if allow_spaces:
        allowed += ", and spaces"

    # Pattern validation
   
    if not pattern.match(text):
        return False, f"Invalid input '{text}'. Allowed characters: {allowed}."

    return True, ""
