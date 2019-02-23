"""
Utility file with functions that handle movie extraction and stemming
"""

import re
import PorterStemmer as ps

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
Takes a list of [title, genres] lists and standardizes entries by extracting the
year and moving the leading article to the end. For example, a single element
conversion: ["The House (2018)", <genres>] -> ["House, The", 2018, <genres>]
"""
def standardize_titles(titles):
    standardized_titles = []

    # Standardize titles: leading article at end, year is extracted
    # For example, "Lottery, The (2016)" -> ["Lottery, The", 2016]
    for entry in titles:
        title = entry[0]
        genres = entry[1]

        # Extract year from title
        title, year = extract_year_from_title(title)

        # Standardize leading article position (move to end)
        title = move_leading_article_to_end(title)

        standardized_titles.append([title, year, genres])

    return standardized_titles

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
