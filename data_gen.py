import random
import csv
import os
from pathlib import Path

# Reproducibility 
random.seed(150)

# Pantry foods 
VEGAN_INGREDIENTS = [
    "rice", "pasta", "lentils", "chickpeas", "black beans", "kidney beans",
    "tofu", "oats", "quinoa", "peanut butter", "almond butter",
    "olive oil", "vegetable broth", "coconut milk", "soy sauce",
    "garlic", "onion", "tomato", "spinach", "broccoli", "carrot",
    "bell pepper", "zucchini", "mushrooms", "sweet potato",
    "frozen peas", "canned tomatoes", "nutritional yeast",
    "lemon juice", "lime", "cumin", "paprika", "turmeric", "chili powder",
    "salt", "black pepper", "oregano", "basil", "thyme",
    "bread", "avocado", "banana", "apple", "blueberries",
    "almond milk", "soy milk", "corn", "celery", "cilantro"
]

VEGETARIAN_EXTRAS = [
    "egg", "milk", "cheese", "butter", "yogurt", "cream cheese",
    "parmesan", "mozzarella", "cheddar", "sour cream",
    "heavy cream", "cottage cheese", "ricotta", "honey"
]

NON_VEG_EXTRAS = [
    "chicken breast", "ground beef", "tuna (canned)", "salmon",
    "shrimp", "turkey", "bacon", "pepperoni", "ham",
    "chicken thighs", "beef broth", "chicken broth", "anchovies",
    "deli turkey", "ground turkey", "pork chop"
]

VEGAN_RECIPES = [
    ("Lentil Soup", 25, "A hearty, warming soup made from pantry staples."),
    ("Black Bean Tacos", 15, "Quick crispy tacos packed with protein."),
    ("Chickpea Curry", 30, "Creamy coconut chickpea curry over rice."),
    ("Pasta Primavera", 20, "Light pasta loaded with fresh vegetables."),
    ("Tofu Stir Fry", 20, "High-protein tofu stir fried with soy sauce and veggies."),
    ("Avocado Toast", 5, "Quick nutritious toast with sliced avocado."),
    ("Quinoa Bowl", 25, "Protein-packed quinoa with roasted vegetables."),
    ("Veggie Fried Rice", 20, "Classic fried rice made entirely vegan."),
    ("Peanut Noodles", 15, "Cold sesame peanut noodles — no cooking required."),
    ("Sweet Potato Curry", 35, "Coconut milk sweet potato curry over rice."),
    ("Oatmeal Bowl", 10, "Hearty oats topped with banana and peanut butter."),
    ("Mushroom Pasta", 25, "Savory garlic mushroom pasta."),
    ("Tomato Basil Soup", 20, "Classic creamy tomato basil soup."),
    ("Rice and Beans", 20, "Simple Latin-inspired rice and black beans."),
    ("Veggie Wrap", 10, "Tortilla wrap stuffed with hummus and fresh veggies."),
    ("Spiced Lentil Dal", 30, "Traditional Indian dal with turmeric and cumin."),
    ("Corn and Bean Salad", 10, "No-cook protein salad with lime dressing."),
    ("Garlic Bread", 10, "Toasted bread with olive oil and garlic."),
    ("Stuffed Bell Peppers", 40, "Bell peppers stuffed with rice and chickpeas."),
    ("Mango Smoothie Bowl", 5, "Blended mango topped with granola and fruit."),
]

VEGETARIAN_RECIPES = [
    ("Cheese Quesadilla", 10, "Golden crispy tortilla with melted cheddar."),
    ("Veggie Omelette", 10, "Fluffy omelette packed with bell pepper and spinach."),
    ("Mac and Cheese", 20, "Classic creamy macaroni and cheese."),
    ("Caprese Salad", 5, "Fresh mozzarella, tomato and basil with olive oil."),
    ("Egg Fried Rice", 15, "Quick fried rice with scrambled egg."),
    ("Pancakes", 15, "Fluffy buttermilk pancakes."),
    ("Greek Yogurt Parfait", 5, "Layered yogurt, granola and fresh berries."),
    ("Spinach and Feta Omelette", 10, "Greek-inspired fluffy omelette."),
    ("Margherita Pizza", 20, "Simple pizza with tomato sauce and mozzarella."),
    ("Scrambled Eggs and Toast", 10, "Classic breakfast scramble with buttered toast."),
    ("Baked Mac and Cheese", 35, "Oven-baked mac with golden breadcrumb topping."),
    ("Cheesy Vegetable Soup", 30, "Thick soup with cheddar and vegetables."),
    ("French Toast", 10, "Egg-soaked bread pan fried with cinnamon."),
    ("Pasta Alfredo", 25, "Fettuccine in a rich parmesan cream sauce."),
    ("Caprese Sandwich", 10, "Fresh mozzarella, tomato and basil on ciabatta."),
    ("Yogurt Smoothie", 5, "Blended yogurt with banana and honey."),
    ("Potato and Egg Scramble", 15, "Hearty potato hash with scrambled egg."),
    ("Cheese Stuffed Mushrooms", 25, "Baked mushrooms filled with cream cheese."),
    ("Ricotta Toast", 5, "Creamy ricotta on toasted bread with honey."),
    ("Vegetable Frittata", 25, "Oven-baked Italian egg frittata."),
]

NON_VEG_RECIPES = [
    ("Chicken Fried Rice", 20, "Classic fried rice with tender chicken pieces."),
    ("Tuna Pasta", 15, "Quick pasta with canned tuna and olive oil."),
    ("Chicken Quesadilla", 15, "Crispy tortilla filled with chicken and cheese."),
    ("Ground Beef Tacos", 20, "Seasoned ground beef in crispy taco shells."),
    ("Shrimp Stir Fry", 20, "Quick shrimp and veggie stir fry with soy sauce."),
    ("BLT Sandwich", 10, "Classic bacon, lettuce and tomato sandwich."),
    ("Chicken Noodle Soup", 30, "Comforting soup with chicken and egg noodles."),
    ("Turkey Wrap", 10, "Deli turkey with cheese and veggies in a tortilla."),
    ("Bacon and Egg Sandwich", 10, "Classic bacon egg and cheese on toast."),
    ("Tuna Salad", 10, "Creamy tuna salad with celery and onion."),
    ("Ground Turkey Bowl", 25, "Seasoned turkey over rice with hot sauce."),
    ("Chicken Caesar Salad", 15, "Grilled chicken over romaine with Caesar dressing."),
    ("Salmon and Rice", 25, "Pan-seared salmon fillet served over steamed rice."),
    ("Pepperoni Pizza", 20, "Classic pizza loaded with mozzarella and pepperoni."),
    ("Chicken Soup", 35, "Simple homemade chicken and vegetable soup."),
    ("Beef and Broccoli", 25, "Takeout-style beef and broccoli stir fry."),
    ("Ham and Cheese Omelette", 10, "Fluffy omelette with ham and melted cheese."),
    ("Shrimp Tacos", 20, "Spiced shrimp tacos with lime slaw."),
    ("Chicken Pasta", 25, "Pasta tossed with grilled chicken and olive oil."),
    ("Anchovy Caesar Salad", 10, "Bold Caesar salad with classic anchovy dressing."),
]


def _pick_ingredients(category: str, n_main: int = None) -> list[str]:
    """
    This function will build a ingredient list for each dietary category. The list must have:
      - All recipes share a common vegan pantry staple.
      - Vegetarian recipes will add dairy/egg extras.
      - Non-vegetarian recipes will add meat/seafood extras.
    """
    n_main = n_main or random.randint(4, 10)

    # Every recipe uses at least some vegan pantry staples
    base = random.sample(VEGAN_INGREDIENTS, min(n_main, len(VEGAN_INGREDIENTS)))

    if category == "Vegetarian":
        extras = random.sample(VEGETARIAN_EXTRAS, random.randint(1, 3))
        base = list(set(base + extras))
    elif category == "Non-Vegetarian":
        meat = random.sample(NON_VEG_EXTRAS, random.randint(1, 2))
        dairy = random.sample(VEGETARIAN_EXTRAS, random.randint(0, 2))
        base = list(set(base + meat + dairy))

    return base[:n_main]


def generate_recipes(n_per_category: int = 70) -> list[dict]:
    """
    Create `n_per_category` recipes for each of the dietary categories
    for a total of 3 × n_per_category records.
    """
    records = []
    recipe_id = 1

    category_map = {
        "Vegan": VEGAN_RECIPES,
        "Vegetarian": VEGETARIAN_RECIPES,
        "Non-Vegetarian": NON_VEG_RECIPES,
    }

    for category, pool in category_map.items():
        for i in range(n_per_category):
            # Cycle through the name pool and use suffix with index to avoid duplicates
            base_name, base_time, base_desc = pool[i % len(pool)]
            suffix = f" #{i // len(pool) + 1}" if i >= len(pool) else ""
            name = base_name + suffix

            # Add small random inconsistencies so prep times are not all similar
            prep_time = max(5, base_time + random.randint(-5, 5))

            ingredients = _pick_ingredients(category)

            records.append({
                "recipe_id": recipe_id,
                "name": name,
                "prep_time_minutes": prep_time,
                "dietary_category": category,
                "cuisine_type": random.choice([
                    "American", "Italian", "Mexican", "Indian",
                    "Asian", "Mediterranean", "Fusion"
                ]),
                "description": base_desc,
                "ingredients": ", ".join(ingredients),
                "num_ingredients": len(ingredients),
                "source": "synthetic",
            })
            recipe_id += 1

    print(f"[data_gen] Generated {len(records)} synthetic recipes "
          f"({n_per_category} per category).")
    return records


def save_to_csv(records: list[dict], output_path: str = "data/synthetic_recipes.csv"):
    """Persist the generated records to a CSV file for later ETL consumption."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(f"[data_gen] Saved to {output_path}")


if __name__ == "__main__":
    recipes = generate_recipes(n_per_category=70)   
    save_to_csv(recipes)
