#!/usr/bin/env python

# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
#
# Usage:
#   python sanity_check.py --recommender
#   python sanity_check.py --binarize
######################################################################
from chatbot import Chatbot


import argparse
import numpy as np
import math


def assertNumpyArrayEquals(givenValue, correctValue, failureMessage):
    try:
        assert np.array_equal(givenValue, correctValue)
        return True
    except Exception:
        print(failureMessage)
        print("Expected: {}".format(correctValue))
        print("Actual: {}".format(givenValue))
        return False

def assertListEquals(givenValue, correctValue, failureMessage, orderMatters=True):
    try:
        if orderMatters:
            assert givenValue == correctValue
            return True
        givenValueSet = set(givenValue)
        correctValueSet = set(correctValue)
        assert givenValueSet == correctValueSet
        return True
    except Exception:
        print(failureMessage)
        print("Expected: {}".format(correctValue))
        print("Actual: {}".format(givenValue))
        return False

def assertEquals(givenValue, correctValue, failureMessage):
    try:
        assert givenValue == correctValue
        return True
    except Exception:
        print(failureMessage)
        print("Expected: {}".format(correctValue))
        print("Actual: {}".format(givenValue))
        return False

def test_similarity():
    print("Testing similarity() functionality...")
    chatbot = Chatbot(False)

    x = np.array([1, 1, -1, 0], dtype=float)
    y = np.array([1, 0, 1, -1], dtype=float)

    self_similarity = chatbot.similarity(x, x)
    if not math.isclose(self_similarity, 1.0):
        print('Unexpected cosine similarity between {} and itself'.format(x))
        print('Expected 1.0, calculated {}'.format(self_similarity))
        print()
        return False

    ortho_similarity = chatbot.similarity(x, y)
    if not math.isclose(ortho_similarity, 0.0):
        print('Unexpected cosine similarity between {} and {}'.format(x, y))
        print('Expected 0.0, calculated {}'.format(ortho_similarity))
        print()
        return False

    print('similarity() sanity check passed!')
    print()
    return True

def test_binarize():
    print("Testing binarize() functionality...")
    chatbot = Chatbot(False)
    if assertNumpyArrayEquals(
        chatbot.binarize(np.array([[1, 2.5, 5, 0]])),
        np.array([[-1., -1., 1., 0.]]),
        "Incorrect output for binarize(np.array([[1, 2.5, 5, 0]]))."
    ):
        print("1. binarize() sanity check passed!")

    if assertNumpyArrayEquals(
        chatbot.binarize(np.array([[0, 1], [2.5, 5]])),
        np.array([[0, -1], [-1, 1]]),
        "Incorrect output for binarize(np.array([[0, 1], [2.5, 5]]))."
    ):
        print("2. binarize() sanity check passed!")
    print()

def test_extract_titles():
    print("Testing extract_titles() functionality...")
    chatbot = Chatbot(False)
    if assertListEquals(
        chatbot.extract_titles('I liked "The Notebook"'),
        ["The Notebook"],
        "Incorrect output for extract_titles(\'I liked \"The Notebook\"\')."
    ) and assertListEquals(
        chatbot.extract_titles('No movies here!'),
        [],
        "Incorrect output for extract_titles('No movies here!').",
    ):
        print('extract_titles() sanity check passed!')
    print()

def test_extract_titles_creative():
    print("Testing extract_titles() creative functionality...")
    chatbot = Chatbot(True)
    if assertListEquals(
        chatbot.extract_titles('I liked the notebook'),
        ["The Notebook"],
        "Incorrect output for extract_titles(\'I liked the notebook\')."
    ) and assertListEquals(
        chatbot.extract_titles('No movies here!'),
        [],
        "Incorrect output for extract_titles('No movies here!').",
    ) and assertListEquals(
        chatbot.extract_titles('I thought 10 things i hate about you was great.'),
        ["10 Things I Hate About You"],
        "Incorrect output for extract_titles('I thought 10 things i hate about you was great.').",
    ) and assertListEquals(
        chatbot.extract_titles('Titanic (1997) started out terrible, but the ending was totally great and I loved it!'),
        ["Titanic (1997)"],
        "Incorrect output for extract_titles('Titanic (1997) started out terrible, but the ending was totally great and I loved it!').",
    ) and assertListEquals(
        chatbot.extract_titles('This undeniable classic is always charming and irresistible, even if far from perfect - the characters in casablanca do not always act consistently with their personalities.'),
        ["Casablanca", 'Always', 'Perfect'],
        "Incorrect output for extract_titles('This undeniable classic is always charming and irresistible, even if far from perfect - the characters in casablanca, for instance, do not always act consistently with their personalities.').",
    ) and assertListEquals(
        chatbot.extract_titles('I liked 10 things i hate about you.'),
        ["10 Things I Hate About You"],
        "Incorrect output for extract_titles('I liked 10 things i hate about you.').",
    ) and assertListEquals(
        chatbot.extract_titles('Titanic (1997), started out terrible, but the ending was totally great and I loved it!'),
        ["Titanic (1997)"],
        "Incorrect output for extract_titles('Titanic (1997), started out terrible, but the ending was totally great and I loved it!').",
    ) and assertListEquals(
        chatbot.extract_titles('I liked 10, things i hate about you.'),
        ["10"],
        "Incorrect output for extract_titles('I liked 10, things i hate about you.').",
    ):
        print('extract_titles() sanity check passed!')
    print()

    """ """

def test_find_movies_by_title():
    print("Testing find_movies_by_title() functionality...")
    chatbot = Chatbot(False)

    if assertListEquals(
        chatbot.find_movies_by_title("The American President"),
        [10],
        "Incorrect output for find_movies_by_title('The American President')."
    ) and assertListEquals(
        chatbot.find_movies_by_title("The AMERICAN President"),
        [],
        "Incorrect output for find_movies_by_title('The AMERICAN President')."
    ) and assertListEquals(
        chatbot.find_movies_by_title("Titanic"),
        [1359, 2716],
        "Incorrect output for find_movies_by_title('Titanic').",
        orderMatters=False
    ) and assertListEquals(
        chatbot.find_movies_by_title("Titanic (1997)"),
        [1359],
        "Incorrect output for find_movies_by_title('Titanic (1997)').",
    ):
        print('find_movies_by_title() sanity check passed!')
    print()

def test_find_movies_by_title_creative():
    print("Testing find_movies_by_title() functionality...")
    chatbot = Chatbot(True)

    if assertListEquals(
        chatbot.find_movies_by_title("The American President"),
        [10],
        "Incorrect output for find_movies_by_title('The American President')."
    ) and assertListEquals(
        chatbot.find_movies_by_title("The AMERICAN President"),
        [10],
        "Incorrect output for find_movies_by_title('The AMERICAN President')."
    ) and assertListEquals(
        chatbot.find_movies_by_title("Titanic"),
        [1359, 2716],
        "Incorrect output for find_movies_by_title('Titanic').",
        orderMatters=False
    ) and assertListEquals(
        chatbot.find_movies_by_title("Titanic (1997)"),
        [1359],
        "Incorrect output for find_movies_by_title('Titanic (1997)').",
    ) and assertListEquals(
        chatbot.find_movies_by_title("Business of"),
        [6924, 3849],
        "Incorrect output for find_movies_by_title('Business of')",
        orderMatters=False
    ) and assertListEquals(
        chatbot.find_movies_by_title("SCREAM"),
        [1142, 1357, 2629, 546],
        "Incorrect output for find_movies_by_title('SCREAM')",
        orderMatters=False
    ) and assertListEquals(
        chatbot.find_movies_by_title("Percy Jackson"),
        [7463, 8377],
        "Incorrect output for find_movies_by_title('Percy Jackson')",
        orderMatters=False
    ) and assertListEquals(
        chatbot.find_movies_by_title("Huang gu shi jie"),
        [792],
        "Incorrect output for find_movies_by_title('Huang gu shi jie')",
        orderMatters=False
    ) and assertListEquals(
        chatbot.find_movies_by_title("Gojira"),
        [1873, 1874, 1875, 3090, 4244, 6070],
        "Incorrect output for find_movies_by_title('Gojira')",
        orderMatters=False
    ) and assertListEquals(
        chatbot.find_movies_by_title("Alive & Kicking"),
        [1306],
        "Incorrect output for find_movies_by_title('Alive & Kicking')",
        orderMatters=False
    )  and assertListEquals(
        chatbot.find_movies_by_title("Las Vampiras"),
        [2585],
        "Incorrect output for find_movies_by_title('Las Vampiras')",
        orderMatters=False
    )  and assertListEquals(
        chatbot.find_movies_by_title("Phantom Love"),
        [2845],
        "Incorrect output for find_movies_by_title('Phantom Love')",
        orderMatters=False
    ):
        print('find_movies_by_title() sanity check passed!')
    print()

def test_extract_sentiment():
    print("Testing extract_sentiment() functionality...")
    chatbot = Chatbot(False)
    if assertEquals(
        chatbot.extract_sentiment("I like \"Titanic (1997)\"."),
        1,
        "Incorrect output for extract_sentiment(\'I like \"Titanic (1997)\".\')"
    ) and assertEquals(
        chatbot.extract_sentiment("I saw \"Titanic (1997)\"."),
        0,
        "Incorrect output for extract_sentiment(\'I saw  \"Titanic (1997)\".\')"
    ) and assertEquals(
        chatbot.extract_sentiment("I didn't enjoy \"Titanic (1997)\"."),
        -1,
        "Incorrect output for extract_sentiment(\'I didn't enjoy  \"Titanic (1997)\"\'.)"
    ):
        print('extract_sentiment() sanity check passed!')
    print()


def test_extract_sentiment_for_movies():
    print("Testing test_extract_sentiment_for_movies() functionality...")
    chatbot = Chatbot(True)
    chatbot.extract_sentiment_for_movies("I liked both \"I, Robot\" and \"Ex Machina\"")
    if assertListEquals(
        chatbot.extract_sentiment_for_movies("I liked both \"I, Robot\" and \"Ex Machina\"."),
        [("I, Robot", 1), ("Ex Machina", 1)],
        "Incorrect output for test_extract_sentiment_for_movies(\"I liked both \"I, Robot\" and \"Ex Machina\".)\"",
        orderMatters=False
    ) and assertListEquals(
        chatbot.extract_sentiment_for_movies("I liked \"I, Robot\" but not \"Ex Machina\"."),
        [("I, Robot", 1), ("Ex Machina", -1)],
        "Incorrect output for test_extract_sentiment_for_movies(\"I liked \"I, Robot\" but not \"Ex Machina\".)\"",
        orderMatters=False
    ) and assertListEquals(
        chatbot.extract_sentiment_for_movies("I liked \"I, Robot\" and \"Lady and the Tramp\", but not \"Ex Machina\"."),
        [("I, Robot", 1), ("Lady and the Tramp", 1), ("Ex Machina", -1)],
        "Incorrect output for test_extract_sentiment_for_movies(\"I liked \"I, Robot\" and \"Lady and the Tramp\", but not \"Ex Machina\".)\"",
        orderMatters=False
    ) and assertListEquals(
        chatbot.extract_sentiment_for_movies("I liked \"I, Robot\", but \"Lady and the Tramp\" was even better! I really liked \"Ex Machina\" too."),
        [("I, Robot", 1), ("Lady and the Tramp", 1), ("Ex Machina", 1)],
        "Incorrect output for test_extract_sentiment_for_movies(\"I liked \"I, Robot\", but \"Lady and the Tramp\" was even better! I really liked \"Ex Machina\" too.\")\"",
        orderMatters=False
    ) and assertListEquals(
        chatbot.extract_sentiment_for_movies("I liked \"I, Robot\", but the \"Lady and the Tramp\" was not good. But I really liked \"Ex Machina\"."),
        [("I, Robot", 1), ("Lady and the Tramp", -1), ("Ex Machina", 1)],
        "Incorrect output for test_extract_sentiment_for_movies(\"I liked \"I, Robot\", but the \"Lady and the Tramp\" was not good. But I really liked \"Ex Machina\".\")\"",
        orderMatters=False
    ) and assertListEquals(
        chatbot.extract_sentiment_for_movies("I liked \"I, Robot\", the \"Lady and the Tramp\", and \"Ex Machina\". \"Metropolitan\", however, was quite disappointing."),
        [("I, Robot", 1), ("Lady and the Tramp", 1), ("Ex Machina", 1), ("Metropolitan", -1)],
        "Incorrect output for test_extract_sentiment_for_movies(\"I liked \"I, Robot\", the \"Lady and the Tramp\", and \"Ex Machina\". \"Metropolitan\", however, was quite disappointing.\")\"",
        orderMatters=False
    ):
        print('extract_sentiment_for_movies() sanity check passed!')
    print()

def test_find_movies_closest_to_title():
    print("Testing find_movies_closest_to_title() functionality...")
    chatbot = Chatbot(True)

    misspelled = "Sleeping Beaty"
    if assertListEquals(
        chatbot.find_movies_closest_to_title(misspelled, max_distance=3),
        [1656],
        "Incorrect output for test_find_movies_closest_to_title('{}', max_distance={})".format(misspelled, 3),
        orderMatters=False
    ):
        print('find_movies_closest_to_title() sanity check passed!')
    print()
    return True

def test_disambiguate():
    print("Testing disambiguate() functionality...")
    chatbot = Chatbot(True)

    clarification1 = "1997"
    candidates1 = [1359, 2716]

    clarification2 = "2"
    candidates2 = [1142, 1357, 2629, 546]

    clarification3 = "sorcerer's stone"
    candidates3 = [3812, 4325, 5399, 6294, 6735, 7274, 7670, 7842]

    clarification4 = "that darn cat!"
    candidates4 = [822, 1090, 1182, 1599]
    # [822]

    clarification5 = "That Darn Cat"
    candidates5 = [822, 1090, 1182, 1599]
    # [1182]

    # Ending punctuation cases
    """
    and assertListEquals(
        chatbot.disambiguate(clarification4, candidates4),
        [822],
        "Incorrect output for disambiguate('{}', {})".format(clarification4, candidates4),
        orderMatters=False
    ) and assertListEquals(
        chatbot.disambiguate(clarification5, candidates5),
        [1182],
        "Incorrect output for disambiguate('{}', {})".format(clarification5, candidates5),
        orderMatters=False
    )
    """

    if assertListEquals(
        chatbot.disambiguate(clarification1, candidates1),
        [1359],
        "Incorrect output for disambiguate('{}', {})".format(clarification1, candidates1),
        orderMatters=False
    ) and assertListEquals(
        chatbot.disambiguate(clarification2, candidates2),
        [1357],
        "Incorrect output for disambiguate('{}', {})".format(clarification2, candidates2),
        orderMatters=False
    ) and assertListEquals(
        chatbot.disambiguate(clarification3, candidates3),
        [3812],
        "Incorrect output for disambiguate('{}', {})".format(clarification3, candidates3),
        orderMatters=False
    ):
        print('disambiguate() sanity check passed!')
    print()
    return True

def test_recommend():
    print("Testing recommend() functionality...")
    chatbot = Chatbot(False)

    user_ratings = np.array([1, -1, 0, 0, 0, 0])
    all_ratings = np.array([
        [1, 1, 1, 0],
        [1, -1, 0, -1],
        [1, 1, 1, 0],
        [0, 1, 1, -1],
        [0, -1, 1, -1],
        [-1, -1, -1, 0],
    ])
    recommendations = chatbot.recommend(user_ratings, all_ratings, 2)

    if assertListEquals(recommendations, [2, 3], "Recommender test failed"):
        print("recommend() sanity check passed!")
    print()

def test_arbitrary_response():
    print("Testing generate_arbitrary_response() functionality...")
    chatbot = Chatbot(True)

    if assertEquals(
        chatbot.generate_arbitrary_response('Can you tell me a story?'),
        "I'm not sure, but I probably can't tell you a story.",
        "Incorrect output for generate_arbitrary_response('Can you tell me a story?')"
    ) and assertEquals(
        chatbot.generate_arbitrary_response('Did you fart really loudly?'),
        "I'm not sure, but I probably didn't fart really loudly.",
        "Incorrect output for generate_arbitrary_response('Did you fart really loudly?')"
    ) and assertEquals(
        chatbot.generate_arbitrary_response('Who is the best actor?'),
        "I don't know who the best actor is.",
        "Incorrect output for generate_arbitrary_response('Who is the best actor?')"
    ):
        print('generate_arbitary_response() sanity check passed!')

def test_process():
    print("Testing process() functionality...")
    chatbot = Chatbot(False)

    user_input = [
    '"Ti"',
    '"Titanic"',
    '"Titanic (1997)"',
    '"Titanic (1997)" I liked',
    '"Titanic (1997)" I disliked',
    '"Avatar" I disliked',
    '"Blade Runner" I enjoyed',
    '"The Notebook" was boring at first but then I think I liked it',
    '"Scream" was bad',
    'I loved "10 Things I Hate About You"',
    'can you recommend me a movie?'
    ]

    print("Testing for simple input")
    for line in user_input:
        print("User: " + line)
        print("Marvin: " + chatbot.process(line))
    print("---------------")

def test_process_creative():
    print("Testing process() functionality...")
    chatbot = Chatbot(True)

    user_input = [
    'I like both "avatar" and "ted 2"',
    'I hated both "avatar" and "ted 2"',
    'I liked "avatar" and I enjoyed "ted 2" as well',
    'I liked "avatar" but I think "ted 2" is really bad',
    'I liked neither "avatar" nor "ted 2"',
    '"avatar" and "ted 2", I liked neither of them',
    'I didn\'t like either "avatar" or "ted 2"',
    'I think "avatar" was good at first but then it got boring. I think "ted 2" was lame at first but then it got worse'
    ]

    print("Testing for multiple movies")
    for line in user_input:
        print("User: " + line)
        print("Marvin: " + chatbot.process(line))
    print("---------------")

def main():
    parser = argparse.ArgumentParser(description='Sanity checks the chatbot. If no arguments are passed, all checks are run; you can use the arguments below to test specific parts of the functionality.')

    parser.add_argument('-b', '--creative', help='Tests all of the creative function', action='store_true')

    args = parser.parse_args()
    testing_creative = args.creative

    test_extract_titles()
    test_find_movies_by_title()
    test_extract_sentiment()
    test_recommend()
    test_binarize()
    test_similarity()
    #test_process()

    if testing_creative:
        test_find_movies_by_title_creative()
        test_find_movies_closest_to_title()
        test_extract_sentiment_for_movies()
        test_disambiguate()
        test_extract_titles_creative()
        #test_process_creative()

if __name__ == '__main__':
    main()
