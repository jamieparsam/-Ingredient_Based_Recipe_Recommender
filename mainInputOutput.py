from recommender_model import RecipeRecommender 


ingredients = input("Enter ingredients, seperated with comma: ").split(",")
ingredients = [i.strip().lower() for i in ingredients]

cookingTime = int(input("Enter max cooking time in minutes (ex 20 means 20 minutes) : "))

dietReq = input("Enter diet (Vegan, Vegetarian, Non-Vegetarian or leave blank for all recipes): ").strip()
if dietReq == "":
    dietReq = None


topDish = int(input("Enter number of recipes you want (ex 3- 5 results): "))


rec = RecipeRecommender()

results = rec.recommend(ingredients, cookingTime, dietReq, topDish)


print("\nRecommended Recipes:\n")

if results.empty:
    print("No recipes found.")
    
else:
    for i in range(len(results)):
        print(results.loc[i, "name"], "-", results.loc[i, "prep_time_minutes"], "mins")