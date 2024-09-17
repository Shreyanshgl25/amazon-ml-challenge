import os
import re
import requests
from PIL import Image
import pytesseract
import pandas as pd

# Dictionary for unit abbreviations and their full forms
unit_abbreviation_map = {
    'cm': 'centimetre',
    'ft': 'foot',
    'in': 'inch',
    'm': 'metre',
    'mm': 'millimetre',
    'yd': 'yard',
    'g': 'gram',
    'kg': 'kilogram',
    'µg': 'microgram',
    'mg': 'milligram',
    'oz': 'ounce',
    'lb': 'pound',
    'ton': 'ton',
    'kv': 'kilovolt',
    'mv': 'millivolt',
    'v': 'volt',
    'kw': 'kilowatt',
    'w': 'watt',
    'cl': 'centilitre',
    'cu ft': 'cubic foot',
    'cu in': 'cubic inch',
    'cup': 'cup',
    'dl': 'decilitre',
    'fl oz': 'fluid ounce',
    'gal': 'gallon',
    'imp gal': 'imperial gallon',
    'l': 'litre',
    'µl': 'microlitre',
    'ml': 'millilitre',
    'pt': 'pint',
    'qt': 'quart'
}

# Mapping of entity units as provided in constants.py
entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon', 'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}

# 1st Function: Extract text from an image
def extract_text_from_image(image_url):
    global i
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
        text = pytesseract.image_to_string(image, lang='eng+fra+deu+spa+ita')
        print(i, end=" ")
        i += 1
        clean_text = text.replace('\n', ' ')
        return clean_text.lower()
    except Exception as e:
        print(f"Error in processing image: {e}")
        return ""

# 2nd Function: Replace unit abbreviations with full forms
def replace_abbreviations_with_optional_space(text):
    for abbr, full_form in unit_abbreviation_map.items():
        pattern = rf'(\d+)\s*{abbr}\b'
        replacement = rf'\1 {full_form}'
        text = re.sub(pattern, replacement, text)
    return text

# 3rd Function: Extract the maximum value with a unit
def extract_max_value_with_unit(text, entity_type):
    units = entity_unit_map.get(entity_type, set())
    if not units:
        return "Invalid entity type or no units found."

    max_value = -float('inf')
    max_unit = None

    for unit in units:
        patterns = [
            rf'(\d+(\.\d+)?)\s*{unit}\b',
            rf'(\d+(\.\d+)?)\s*{unit}s?\b',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    value = float(match[0])
                    if value > max_value:
                        max_value = value
                        max_unit = unit
                except ValueError:
                    continue

    if max_unit:
        return f"{max_value} {max_unit}"
    else:
        return ""

# Predictor function to integrate all 3 functions
def predictor(image_link, category_id, entity_name):
    # 1. Extract text from image
    extracted_text = extract_text_from_image(image_link)

    # 2. Replace abbreviations with full forms
    extracted_text = replace_abbreviations_with_optional_space(extracted_text)

    # 3. Extract the maximum value with a unit
    prediction = extract_max_value_with_unit(extracted_text, entity_name)
    
    return prediction

# Main block for reading dataset and applying the predictor
if __name__ == "__main__":
    DATASET_FOLDER = '../dataset/'

    # Load the dataset
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))

    # Apply the predictor function
    test['prediction'] = test.apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1)

    # Save the output
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)