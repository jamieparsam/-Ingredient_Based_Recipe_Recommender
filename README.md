# -Ingredient_Based_Recipe_Recommender
CS 210 Project 




### Installation process:
2 ways of installing:

# Clone the repo 
git clone <https://github.com/jamieparsam/-Ingredient_Based_Recipe_Recommender.git> 

# Zip code download
Click <Code> and download the entire repo as Zip File, also will be provided





### Running the Code

# 1) Run ETL pipeline (IMPORTANT)
Start with running the ETL Pipeline using this code:
python3 etl_pipeline.py

This creates a cleaned_recipe CSV file from a mix of Kaggle and Synthetic dataset. 
*Required for running the Main file

# 2) Run the recommendation system 
Run recommandation system using the mainInputOutput.py file
python3 mainInputOutput.py

It asks you for prompts such as:
Ingredients: Write ingridents and seperate them with comma, EX: bread, milk, cheese
Cooking Max Time: Write whats the longest amount of time you can spend cooking, EX: 10-25, represents MIN
Diet: Write either of the 3, if any 3 dietary requirement work then enter nothing [ SPELLING IS IMPORTANT ]
Top N: This means how many recipes you want that follows your requirement the most, EX: 3 means 3 differnt recipes that matches your preferences

Lastly you get the output which showcases the meal and how long it takes to prepare it.













