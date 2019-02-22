"""Utility methods to load movie data from data files.

Ported to Python 3 by Matt Mistele (@mmistele) and Sam Redmond (@sredmond).

Intended for PA6 in Stanford's Winter 2019 CS124.
"""
import csv
import pathlib
import re

import numpy as np

ME = pathlib.Path(__file__).parent

DATA_FOLDER = ME / 'data'

RATINGS_FILE = str(DATA_FOLDER / 'ratings.txt')
MOVIES_FILE = str(DATA_FOLDER / 'movies.txt')
SENTIMENT_FILE = str(DATA_FOLDER / 'sentiment.txt')


def ratings(src_filename=RATINGS_FILE, delimiter='%', header=False, quoting=csv.QUOTE_MINIMAL):
    title_list = titles()
    user_id_set = set()
    with open(src_filename, 'r') as f:
        content = f.readlines()
        for line in content:
            user_id = int(line.split(delimiter)[0])
            if user_id not in user_id_set:
                user_id_set.add(user_id)
    num_users = len(user_id_set)
    num_movies = len(title_list)
    mat = np.zeros((num_movies, num_users))

    with open(src_filename) as f:
        reader = csv.reader(f, delimiter=delimiter, quoting=quoting)
        if header:
            next(reader)
        for line in reader:
            mat[int(line[1])][int(line[0])] = float(line[2])
    return title_list, mat

# Extracts an year from the input title, returns ([title without year], year) 
# if an year was successfully extracted, and (title, None) otherwise
def extract_year_from_title(title):
    year_pat = '^(?P<title>.*) \((?P<year>[0-9]{4}|[0-9]{4}-|[0-9]{4}-[0-9]{4})\)$'

    e_matches = re.findall(year_pat, title)
    if (len(e_matches) != 1):
        # no year found
        movie_title = title
        movie_year = None
    else:
        # successfully parsed year
        assert (len(e_matches) == 1)
        movie_title = e_matches[0][0]
        movie_year = e_matches[0][1]

    return movie_title, movie_year

# Standardizes titles by moving the leading article to the end of the title
# For example, "The Blue Dog" -> "Blue Dog, The"
# NOTE: does not discriminate on year (i.e. extract year before using this
# function); "An Apple (2010)" -> "Apple (2010), An"
def move_leading_article_to_end(title):
    # matches "<article> <title>"
    # for example, matches "The Notebook"
    article_pat = '^(?P<leading_article>A|a|An|an|AN|The|the|THE) (?P<title>.*)$'

    # check for leading article
    e_matches = re.findall(article_pat, title)
    if len(e_matches) != 0:
        # leading article detected
        assert (len(e_matches) == 1)

        movie_article = e_matches[0][0]
        movie_title = e_matches[0][1]

        return '{}, {}'.format(movie_title, movie_article)
    else:
        return title


def titles(src_filename=MOVIES_FILE, delimiter='%', header=False, quoting=csv.QUOTE_MINIMAL):
    with open(src_filename) as f:
        reader = csv.reader(f, delimiter=delimiter, quoting=quoting)
        if header:
            next(reader)

        title_list = []
        for line in reader:
            movieID, title, genres = int(line[0]), line[1], line[2]
            if title[0] == '"' and title[-1] == '"':
                title = title[1:-1]

            # Standardize titles: leading article at end, year is extracted
            # For example, "Lottery, The (2016)" -> ["Lottery, The", 2016]

            # Extract year from title
            title, year = extract_year_from_title(title)

            # Standardize leading article position (move to end)
            title = move_leading_article_to_end(title)

            title_list.append([title, year, genres])
    return title_list


def sentiment(src_filename=SENTIMENT_FILE, delimiter=',', header=False, quoting=csv.QUOTE_MINIMAL):
    with open(src_filename, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter, quoting=quoting)
        if header:
            next(reader)  # Skip the first line.
        return dict(reader)
