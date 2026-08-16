# nutrient_data.py
nutrient_info = {
    "apple": {"calories": 52, "fiber": 2.4, "vitamin_c": 4.6},
    "banana": {"calories": 96, "fiber": 2.6, "vitamin_c": 8.7},
    "orange": {"calories": 43, "fiber": 2.2, "vitamin_c": 53.2},
    "granny_smith": {"calories": 52, "Carbs": 14, "Protein": 0.3, "Fat": 0.2},
    "ester apple": {"calories": 50, "Carbs": 13, "Protein": 0.2, "Fat": 0.1},
    "strawberry": {"calories": 32, "fiber": 2, "Protein": 0.7, "Fat": 0.3},
    "pineapple": {"calories": 50, "fiber": 0.1, "Protein": 0.5, "Fat": 0.1},
    # Add more fruits here...
}


def get_nutrient_info(fruit_name):
    return nutrient_info.get(fruit_name.lower(), "Nutrient info not available")
