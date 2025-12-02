import re
import inspect
import streamlit as st

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


def check_input_validation(text: str, mode: str, allow_spaces: bool = False, print_errors:bool = False, error_suffix:str | None = None) -> tuple[bool, str]:
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

    func_name = inspect.currentframe().f_code.co_name

    # Basic checks
    if not text or text.strip() == "":
        if error_suffix:
            error_message = f"{error_suffix}: Input cannot be empty."
        else:
            error_message = "Input cannot be empty."

        if print_errors:
            st.error(error_message)
        return False, error_message
    
    if len(text.strip()) < 4:
        if error_suffix:
            error_message = f"{error_suffix}: Input must be at least 4 characters long."
        else:
            error_message = "Input must be at least 4 characters long."
        if print_errors:
            st.error(error_message)
        return False, error_message

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
            f"{func_name}: Invalid mode: '{mode}'. Must be one of: "
            f"'{MODE_CHARS_NUMS}', '{MODE_ONLY_CHARS}', or '{MODE_ONLY_NUMS}'."
        )

    if allow_spaces:
        allowed += ", and spaces"

    # Pattern validation
   
    if not pattern.match(text):
        if error_suffix:
            error_message = f"{error_suffix}: Invalid input '{text}'. Allowed characters: {allowed}."
        else:
            error_message = f"Invalid input '{text}'. Allowed characters: {allowed}."
        if print_errors:
            st.error(error_message)
        return False, error_message

    return True, ""
