def validate_case(data, default_case):
    """
    Ensures the structured data:
    - has correct types
    - has valid values
    - normalizes values
    
    IMPORTANT: 
    - Only include keys that exist in the input data OR are explicitly None
    - Missing fields (None) are preserved, NOT filled with defaults
    - This is critical for the non-assumptive policy evaluation approach
    """

    validated = {}
    errors = []

    # Only process keys that exist in the input data
    for key in data.keys():
        if key not in default_case:
            continue
            
        default_value = default_case[key]
        value = data.get(key)
        
        # -----------------------------
        # PRESERVE NULL VALUES
        # -----------------------------
        if value is None:
            validated[key] = None
            continue

        # -----------------------------
        # TYPE CHECKING AND CONVERSION
        # -----------------------------
        if isinstance(default_value, bool):
            if isinstance(value, bool):
                validated[key] = value
            elif isinstance(value, str):
                # Try to parse string boolean
                if value.lower() in ('true', 'false'):
                    validated[key] = value.lower() == 'true'
                else:
                    validated[key] = None
            else:
                validated[key] = None

        elif isinstance(default_value, int):
            try:
                validated[key] = int(value)
            except (ValueError, TypeError):
                validated[key] = None

        elif isinstance(default_value, str):
            try:
                validated[key] = str(value).lower()
            except:
                validated[key] = None

    return validated, errors