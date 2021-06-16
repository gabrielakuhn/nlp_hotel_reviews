#Separating Positive_Review e Negative_Review in two different txt files... it's take a while to finish
import pandas as pd

DATASET_PATH = "dataset/hotel_reviews.csv";

df = pd.read_csv(DATASET_PATH)

positive_reviews = df["Positive_Review"]
negative_reviews = df["Negative_Review"]

#txt file for Positive_Review
for positive_review in positive_reviews:
    with open("positive_reviews.txt", "a") as fileObject:
        if positive_review != "No Positive":
            print(positive_review, "positive_review")
            fileObject.write(positive_review.lower())
            fileObject.write("\n")

#txt file for Negative_Review
for negative_review in negative_reviews:
    with open("negative_reviews.txt", "a") as fileObject:
        if negative_review != "No Negative":
            print(negative_review, "negative_review")
            fileObject.write(negative_review.lower())
            fileObject.write("\n")

