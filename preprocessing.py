import csv
import random
from sklearn.model_selection import train_test_split
import re


def preprocessData():
    # Headers: Show Number, Air Date, Round, Category, Value, Question, Answer
    # For now, just extracting the question and the round

    questions = []
    rounds = []

    with open("JEOPARDY_CSV.csv", encoding="utf8") as csvfile:
        # Initialize csv reader
        reader = csv.reader(csvfile)

        # Skip headers
        next(reader)

        # Append questions and the round for each to the respective lists
        for row in reader:
            if row[2] != "Final Jeopardy!":  # Disregard any Final Jeopardy rounds
                questions.append(re.sub('<[^>]+>', '', row[5]))
                rounds.append(row[2])

    # Give rounds a number label if they are from first or second round
    rounds = [0 if round == 'Jeopardy!' else 1 for round in rounds]

    # Shuffle both lists together by first zipping them, and then shuffling
    temp = list(zip(questions, rounds))
    random.Random(329).shuffle(temp)
    questions, rounds = zip(*temp)

    # Split data into train, test and dev (can be further split if needed)
    x_train, x_test, y_train, y_test = train_test_split(questions, rounds, test_size=0.2, random_state=329)

    return x_train, x_test, y_train, y_test
