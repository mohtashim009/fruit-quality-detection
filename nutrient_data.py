# nutrient_data.py

NUTRIENT_DATA = {
    "apple": {
        "calories": 52,
        "fiber": 2.4,
        "vitamin_c": 4.6
    },

    "banana": {
        "calories": 89,
        "fiber": 2.6,
        "vitamin_c": 8.7
    },

    "bellpepper": {
        "calories": 31,
        "fiber": 2.1,
        "vitamin_c": 127.7
    },

    "carrot": {
        "calories": 41,
        "fiber": 2.8,
        "vitamin_c": 5.9
    },

    "cucumber": {
        "calories": 15,
        "fiber": 0.5,
        "vitamin_c": 2.8
    },

    "grape": {
        "calories": 69,
        "fiber": 0.9,
        "vitamin_c": 10.8
    },

    "guava": {
        "calories": 68,
        "fiber": 5.4,
        "vitamin_c": 228.3
    },

    "jujube": {
        "calories": 79,
        "fiber": 10.0,
        "vitamin_c": 69
    },

    "mango": {
        "calories": 60,
        "fiber": 1.6,
        "vitamin_c": 36.4
    },

    "orange": {
        "calories": 47,
        "fiber": 2.4,
        "vitamin_c": 53.2
    },

    "pomegranate": {
        "calories": 83,
        "fiber": 4.0,
        "vitamin_c": 10.2
    },

    "potato": {
        "calories": 77,
        "fiber": 2.2,
        "vitamin_c": 19.7
    },

    "strawberry": {
        "calories": 32,
        "fiber": 2.0,
        "vitamin_c": 58.8
    },

    "tomato": {
        "calories": 18,
        "fiber": 1.2,
        "vitamin_c": 13.7
    }
}

def get_nutrient_info(fruit_name):
    """
    Return nutritional information for the detected fruit.
    """
    fruit_name = fruit_name.lower().strip()

    return NUTRIENT_DATA.get(fruit_name)