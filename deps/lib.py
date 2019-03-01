"""
Utility file with functions that handle movie extraction and stemming
"""

import re
import random
import PorterStemmer as ps

## Movie title extraction helpers ##

"""
Extracts the year from the input title, returns ([title without year], year)
if an year was successfully extracted, and (title, None) otherwise
"""
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

"""
Standardizes titles by moving the leading article to the end of the title
For example, "The Blue Dog" -> "Blue Dog, The"
NOTE: does not discriminate on year (i.e. extract year before using this
function); "An Apple (2010)" -> "Apple (2010), An"
"""
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


"""
Standardizes titles by moving the leading article to the front of the title
For example, "Blue Dog, The" -> "The Blue Dog"
NOTE: does not discriminate on year (i.e. extract year before using this
function); "Apple, An (2010)" -> "Apple, An (2010)"
"""
def move_leading_article_to_front(title):
    article_pat = '^(?P<title>.*), (?P<leading_article>A|a|An|an|AN|The|the|THE)$'

    # check for leading article
    e_matches = re.findall(article_pat, title)
    if len(e_matches) != 0:
        # leading article detected at end of title
        assert (len(e_matches) == 1)

        movie_title = e_matches[0][0]
        movie_article = e_matches[0][1]

        return '{} {}'.format(movie_article, movie_title)
    else:
        return title

"""
Takes in a list of movie title objects(self.titles) and movie indicies
Returns a list of string type movie names with year included
"""
def extract_movies_using_indices(titles_obj, indices):
    movies = []
    for index in indices:
        title, year, genre = titles_obj[index]
        if year is not None:
            title += " (" + str(year) + ")"
            movies.append(title)
    return movies

"""
Takes a list of [title, genres] lists and standardizes entries by extracting the
year and moving the leading article to the front. For example, a single element
conversion: ["House, The (2018)", <genres>] -> ["The House", 2018, <genres>]
"""
def standardize_titles(titles):
    standardized_titles = []

    # Standardize titles: leading article at front, year is extracted
    # For example, "Lottery, The (2016)" -> ["The Lottery", 2016]
    for entry in titles:
        title = entry[0]
        genres = entry[1]

        # Extract year from title
        title, year = extract_year_from_title(title)

        # Standardize leading article position (move to front)
        title = move_leading_article_to_front(title)

        standardized_titles.append([title, year, genres])

    return standardized_titles

"""
Used for extracting titles from user input when no quotation marks are used.
Attempts to match movie_title to input_text word-by-word using a 'common
substring' algorithm (words don't need to be consecutive)
"""
def extract_title_by_word(movie_title, input_text):
    mt_tks = movie_title.split()
    it_tks = input_text.split()

    mt_index = 0

    for i in range(len(it_tks)):
        it_word = it_tks[i]
        mt_word = mt_tks[mt_index]


        # Attempts to match last word of title in the case where input has ending punctuation
        if mt_index == (len(mt_tks) - 1):
            if not it_word[-1].isalnum() and it_word[-1] != mt_word[-1]:
                it_word = it_word[:-1]

        if it_word == mt_word:
            mt_index += 1

        if mt_index == len(mt_tks):
            return True

    return False

"""
Used to determine if a string is a word-by-word substring of the movie title
Takes in the substring to check and a movie title
Returns True if the substring is found inside the movie title and False otherwise
Empty string should return False

Example:
title_contains_words("scream", "Scream") --> True
title_contains_words("scream", "Scream 2") --> True
title_contains_words("scream", "Screaming") --> False
title_contains_words("", "Some movie") --> False
"""
def title_contains_words(substring, movie_title):
    if substring == "":
        return False
    substring_pos = movie_title.lower().find(substring.lower())
    substring_end_pos = substring_pos + len(substring)
    # If a substring match exists, and the match is at the end of the title or the title character right after the match is not a letter, return True
    # A similar check is applied to the begining of the match so there are no extra prefix letters
    if substring_pos != -1 and (substring_end_pos == len(movie_title) or not movie_title[substring_end_pos].isalpha()) \
            and (substring_pos == 0 or not movie_title[substring_pos - 1].isalpha()):
        return True

    return False

"""
Calculates the minimum edit distance between w1 and w2.
Insertions, deletions have cost of 1, and replacements have cost of 2.
"""
cache = {}
def min_edit_distance(w1, w2):
    cache = {}
    return min_edit_distance_helper(w1, w2, len(w1)-1, len(w2)-1, cache)

def min_edit_distance_helper(w1, w2, i1, i2, cache):
    # base case
    if i1 == -1:
        return i2 + 1
    if i2 == -1:
        return i1 + 1

    # recursive cases
    if w1[i1] == w2[i2]:
        k = (i1-1, i2-1)
        if k not in cache:
            ed = min_edit_distance_helper(w1, w2, i1-1, i2-1, cache)
        else:
            ed = cache[k]
    else:
        k1 = (i1-1, i2)
        if k1 not in cache:
            ed1 = min_edit_distance_helper(w1, w2, i1-1, i2, cache) + 1
        else:
            ed1 = cache[k1] + 1

        k2 = (i1, i2-1)
        if k2 not in cache:
            ed2 = min_edit_distance_helper(w1, w2, i1, i2-1, cache) + 1
        else:
            ed2 = cache[k2] + 1

        ed = min(ed1, ed2)

    k = (i1, i2)
    cache[k] = ed

    return ed

## Movies sentiment extraction helpers ##

"""
Tokenize text to separate into segments (tokens) of conjunctions,
movie titles, and other words.
Note: assumes that all words within "" are movie titles
Returns list of [(<tag>, <token>), (<tag>, <token>), ...]
"""
def tokenize_conj_movie_other (text):
    # Coordinating conjunctions: for, and, nor, but, or, yet, so
    # conjunctions = {'for', 'and', 'nor', 'but', 'or', 'yet', 'so'}
    # or, nor, and -> same clause
    # but -> diff clause
    conjunctions = {'and', 'nor', 'but', 'or'}

    TKN_TITLE = 0
    TKN_CONJ = 1
    TKN_OTHER = 2

    processed_tokens = []

    tokens = text.split('\"')
    i = 0
    while (i < len(tokens)):
        assert (i % 2 == 0)

        # token[i] is a non-movie segment of words
        word_segment = tokens[i]

        # tokenize by conjunctions, i.e. create tokens of
        # [<non-conj-words>, <conj>, <non-conj-words>, ...]
        word_tokens = word_segment.split()
        string_builder = []
        conj_tokenized_words = []

        for word in word_tokens:
            if word.lower() not in conjunctions:
                # Non-conjunction word, continue building word token
                string_builder.append(word)
            else:
                # Word is a conjunction
                conj_tokenized_words.append((TKN_OTHER, ' '.join(string_builder)))
                conj_tokenized_words.append((TKN_CONJ, word))
                string_builder = []

        if len(string_builder) > 0:
            conj_tokenized_words.append((TKN_OTHER, ' '.join(string_builder)))

        processed_tokens = processed_tokens + conj_tokenized_words

        if (i + 1) < len(tokens):
            movie_title = tokens[i + 1]
            processed_tokens.append((TKN_TITLE, movie_title))

        i += 2

    # Remove all empty tokens
    tagged_tokens = []
    for elem in processed_tokens:
        (_, token) = elem
        if token != '':
            tagged_tokens.append(elem)

    return tagged_tokens

## Stemming helpers##

"""
Takes in a text string and returns the text string stemmed using PorterStemmer
"""
def stem_text(text):
  stemmer = ps.PorterStemmer()
  output = ''
  word = ''
  for c in text:
      if c.isalpha():
          word += c.lower()
      else:
          if word:
              output += stemmer.stem(word, 0,len(word)-1)
              word = ''
          output += c.lower()
  output += stemmer.stem(word, 0,len(word)-1)
  return output

"""
Takes a map from words to 'pos' 'neg' labels and updates the keys into its stemmed
versions. (self.sentiment in chatbot). Returns the stemmed version of that map.
Stemming is done using PorterStemmer
{'enjoy' : 'pos'} -> {'enjoi' : 'pos'}
"""
def stem_map(mapping):
    result = {}
    for key, val in mapping.items():
        result[stem_text(key)] = val

    return result

## Response helpers ##
"""
Takes in a corpus, which is a list of potential responces.
Returns one of the responses in the corpus with uniform probability
"""
def getResponse(corpus):
    index = random.randint(0, len(corpus) - 1)
    return corpus[index]

"""
Concatenates a list of titles gramatically using the specified conjuction
"""
def concatenate_titles(titles, final_conj):
    if len(titles) == 1:
        return titles[0]
    if len(titles) == 2:
        return titles[0] + " " + final_conj + " " + titles[1]
    if len(titles) > 2:
        return ', '.join(titles[0:-1]) + ', ' + final_conj + ' ' + titles[-1]
