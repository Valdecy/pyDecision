###############################################################################

# Required Libraries
import re

###############################################################################

def extract_number(text):
    """
    Extract the first integer from a text string.
    
    Parameters
    ----------
    text : str
        Input string potentially containing a rank or numeric value.
    
    Returns
    -------
    int or None
        The first integer found in the string, or None if no digit is present.
    """
    match = re.search(r'\d+', text)
    return int(match.group()) if match else None

###############################################################################
