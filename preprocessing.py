import csv
import random
from sklearn.model_selection import train_test_split

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
            questions.append(row[5])
            rounds.append(row[2])

    # Remove any Final Jeopardy rounds to focus only on 1st and 2nd round
    while "Final Jeopardy!" in rounds:
        rounds.remove("Final Jeopardy!")

    # Give rounds a number label if they are from first or second round
    rounds = [1 if round == 'Jeopardy!' else 2 for round in rounds]

    # Shuffle both lists together by first zipping them, and then shuffling
    temp = list(zip(questions, rounds))
    random.Random(329).shuffle(temp)
    questions, rounds = zip(*temp)

    # Split data into train, test and dev
    x_train, x_testanddev, y_train, y_testanddev = train_test_split(questions, rounds, test_size=0.3, random_state=329)
    x_test, x_dev, y_test, y_dev = train_test_split(x_testanddev, y_testanddev, test_size=0.5, random_state=329)

    return x_train, x_test, x_dev, y_train, y_test, y_dev
