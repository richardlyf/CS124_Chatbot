# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import movielens

import numpy as np
import re

from deps import sentimentPrediction as sentiPred
from deps import lib


class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
      # The chatbot's default name is `moviebot`. Give your chatbot a new name.
      self.name = 'moviebot'

      self.creative = creative

      # This matrix has the following shape: num_movies x num_users
      # The values stored in each row i and column j is the rating for
      # movie i by user j
      self.titles, ratings = movielens.ratings()
      self.titles = lib.standardize_titles(self.titles)

      self.sentiment = movielens.sentiment()
      self.sentiment = lib.stem_map(self.sentiment)

      #############################################################################
      # TODO: Binarize the movie ratings matrix.                                  #
      #############################################################################

      # Binarize the movie ratings before storing the binarized matrix.
      self.ratings = ratings
      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################

    #############################################################################
    # 1. WARM UP REPL                                                           #
    #############################################################################

    def greeting(self):
      """Return a message that the chatbot uses to greet the user."""
      #############################################################################
      # TODO: Write a short greeting message                                      #
      #############################################################################

      greeting_message = "How can I help you?"

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return greeting_message

    def goodbye(self):
      """Return a message that the chatbot uses to bid farewell to the user."""
      #############################################################################
      # TODO: Write a short farewell message                                      #
      #############################################################################

      goodbye_message = "Have a nice day!"

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return goodbye_message


    ###############################################################################
    # 2. Modules 2 and 3: extraction and transformation                           #
    ###############################################################################

    def process(self, line):
      """Process a line of input from the REPL and generate a response.

      This is the method that is called by the REPL loop directly with user input.

      You should delegate most of the work of processing the user's input to
      the helper functions you write later in this class.

      Takes the input string from the REPL and call delegated functions that
        1) extract the relevant information, and
        2) transform the information into a response to the user.

      Example:
        resp = chatbot.process('I loved "The Notebok" so much!!')
        print(resp) // prints 'So you loved "The Notebook", huh?'

      :param line: a user-supplied line of text
      :returns: a string containing the chatbot's response to the user input
      """
      #############################################################################
      # TODO: Implement the extraction and transformation in this method,         #
      # possibly calling other functions. Although modular code is not graded,    #
      # it is highly recommended.                                                 #
      #############################################################################
      if self.creative:
        response = "I processed {} in creative mode!!".format(line)
      else:
        response = "I processed {} in starter mode!!".format(line)
        self.extract_sentiment(line)

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return response

    def extract_titles(self, text):
      """Extract potential movie titles from a line of text.

      Given an input text, this method should return a list of movie titles
      that are potentially in the text.

      - If there are no movie titles in the text, return an empty list.
      - If there is exactly one movie title in the text, return a list
      containing just that one movie title.
      - If there are multiple movie titles in the text, return a list
      of all movie titles you've extracted from the text.

      Example:
        potential_titles = chatbot.extract_titles('I liked "The Notebook" a lot.')
        print(potential_titles) // prints ["The Notebook"]

      :param text: a user-supplied line of text that may contain movie titles
      :returns: list of movie titles that are potentially in the text
      """
      titles = []

      quote_pat = '\"(?P<title>[^\""]+)\"'
      e_matches = re.findall(quote_pat, text)
      for title in e_matches:
        titles.append(title)

      return titles

    # If max_distance is provided, then enable max-distance spell correcting.
    # Note that the spell correcting is applied after year extraction.
    def find_movies_by_title(self, title, max_distance=-1):
      """ Given a movie title, return a list of indices of matching movies.

      - If no movies are found that match the given title, return an empty list.
      - If multiple movies are found that match the given title, return a list
      containing all of the indices of these matching movies.
      - If exactly one movie is found that matches the given title, return a list
      that contains the index of that matching movie.

      Example:
        ids = chatbot.find_movies_by_title('Titanic')
        print(ids) // prints [1359, 1953]

      :param title: a string containing a movie title
      :returns: a list of indices of matching movies
      """
      movies = []

      movie_title = title
      movie_year = None

      # Extract year from title
      movie_title, movie_year = lib.extract_year_from_title(title)

      # Standardize leading article position (move to front)
      movie_title = lib.move_leading_article_to_front(movie_title)

      # Search through movie title database for matching titles
      for i in range(len(self.titles)):
        entry_title, entry_year, _ = self.titles[i]

        # Filter by movie year
        if movie_year is not None and movie_year != entry_year:
          continue

        # Filter by movie title
        if max_distance <= 0:
          # No spelling correction
          if movie_title.lower() == entry_title.lower():
            movies.append(i)
        else:
          # Apply spelling correction
          if lib.min_edit_distance(movie_title.lower(), entry_title.lower()) <= max_distance:
            movies.append(i)

      return movies


    def extract_sentiment(self, text):
      """Extract a sentiment rating from a line of text.

      You should return -1 if the sentiment of the text is negative, 0 if the
      sentiment of the text is neutral (no sentiment detected), or +1 if the
      sentiment of the text is positive.

      As an optional creative extension, return -2 if the sentiment of the text
      is super negative and +2 if the sentiment of the text is super positive.

      Example:
        sentiment = chatbot.extract_sentiment('I liked "The Titanic"')
        print(sentiment) // prints 1

      :param text: a user-supplied line of text
      :returns: a numerical value for the sentiment of the text
      """
      # Naive implementation of just counting positive words vs negative words
      negations = ['don\'t', 'not', 'never', 'none', 'nothing', 'hardly', 'didn\'t', 'but']
      negation_scale = 1

      words_stemmed = lib.stem_text(text)
      words = words_stemmed.split()
      posCount = 0
      print(words)
      for word in words:
          # All words after the negation will take on its inverted value
          if word in negations:
              negation_scale *= -1

          # Reset the negation scale to 1 if at end punctuation
          if not word[len(word) - 1].isalpha():
              negation_scale = 1

          if word in self.sentiment:
              if self.sentiment[word] == 'pos':
                  print("Word: " + word + " score: " + str(negation_scale))
                  posCount += negation_scale
              else:
                  print("Word: " + word + " score: " + str(-negation_scale))
                  posCount -= negation_scale

      if posCount != 0:
          posCount = posCount / abs(posCount)
      print(posCount)
      return posCount

      # Should transfer over the Naive Bayes probabilities and try those

      '''
      # This is the logistic regression frame work. Need bigram features for this to work

      words_stemmed = lib.stem_text(text)
      words = text.split()

      classifier = sentiPred.SentimentPredictor(self.sentiment)

      return classifier.classify(words_stemmed)
      '''

    def extract_sentiment_for_movies(self, text):
      """Creative Feature: Extracts the sentiments from a line of text
      that may contain multiple movies. Note that the sentiments toward
      the movies may be different.

      You should use the same sentiment values as extract_sentiment, described above.
      Hint: feel free to call previously defined functions to implement this.

      Example:
        sentiments = chatbot.extract_sentiment_for_text('I liked both "Titanic (1997)" and "Ex Machina".')
        print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

      :param text: a user-supplied line of text
      :returns: a list of tuples, where the first item in the tuple is a movie title,
        and the second is the sentiment in the text toward that movie
      """
      # Coordinating conjunctions: for, and, nor, but, or, yet, so
      conjunctions = {'for', 'and', 'nor', 'but', 'or', 'yet', 'so'}

      # Enums for tracking token types
      TKN_TITLE = 0
      TKN_CONJ = 1
      TKN_OTHER = 2

      # Tokenize text to separate into segments (tokens) of conjunctions,
      # movie titles, and other words
      # Note: assume that all words within "" are movie titles
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

      print(tagged_tokens)
      pass

    def find_movies_closest_to_title(self, title, max_distance=3):
      """Creative Feature: Given a potentially misspelled movie title,
      return a list of the movies in the dataset whose titles have the least edit distance
      from the provided title, and with edit distance at most max_distance.

      - If no movies have titles within max_distance of the provided title, return an empty list.
      - Otherwise, if there's a movie closer in edit distance to the given title
        than all other movies, return a 1-element list containing its index.
      - If there is a tie for closest movie, return a list with the indices of all movies
        tying for minimum edit distance to the given movie.

      Example:
        chatbot.find_movies_closest_to_title("Sleeping Beaty") # should return [1656]

      :param title: a potentially misspelled title
      :param max_distance: the maximum edit distance to search for
      :returns: a list of movie indices with titles closest to the given title and within edit distance max_distance
      """
      return self.find_movies_by_title(title, max_distance)


    def disambiguate(self, clarification, candidates):
      """Creative Feature: Given a list of movies that the user could be talking about
      (represented as indices), and a string given by the user as clarification
      (eg. in response to your bot saying "Which movie did you mean: Titanic (1953)
      or Titanic (1997)?"), use the clarification to narrow down the list and return
      a smaller list of candidates (hopefully just 1!)

      - If the clarification uniquely identifies one of the movies, this should return a 1-element
      list with the index of that movie.
      - If it's unclear which movie the user means by the clarification, it should return a list
      with the indices it could be referring to (to continue the disambiguation dialogue).

      Example:
        chatbot.disambiguate("1997", [1359, 2716]) should return [1359]

      :param clarification: user input intended to disambiguate between the given movies
      :param candidates: a list of movie indices
      :returns: a list of indices corresponding to the movies identified by the clarification
      """
      pass


    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################

    def binarize(self, ratings, threshold=2.5):
      """Return a binarized version of the given matrix.

      To binarize a matrix, replace all entries above the threshold with 1.
      and replace all entries at or below the threshold with a -1.

      Entries whose values are 0 represent null values and should remain at 0.

      :param x: a (num_movies x num_users) matrix of user ratings, from 0.5 to 5.0
      :param threshold: Numerical rating above which ratings are considered positive

      :returns: a binarized version of the movie-rating matrix
      """
      binarized_ratings = ratings.copy()
      binarized_ratings[np.where((binarized_ratings <= threshold) & (binarized_ratings != 0))] = -1
      binarized_ratings[np.where(binarized_ratings > threshold)] = 1

      return binarized_ratings


    def similarity(self, u, v):
      """Calculate the cosine similarity between two vectors.

      You may assume that the two arguments have the same shape.

      :param u: one vector, as a 1D numpy array
      :param v: another vector, as a 1D numpy array

      :returns: the cosine similarity between the two vectors
      """
      return np.dot (u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
      """Generate a list of indices of movies to recommend using collaborative filtering.

      You should return a collection of `k` indices of movies recommendations.

      As a precondition, user_ratings and ratings_matrix are both binarized.

      Remember to exclude movies the user has already rated!

      :param user_ratings: a binarized 1D numpy array of the user's movie ratings
      :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
        `ratings_matrix[i, j]` is the rating for movie i by user j
      :param k: the number of recommendations to generate
      :param creative: whether the chatbot is in creative mode

      :returns: a list of k movie indices corresponding to movies in ratings_matrix,
        in descending order of recommendation
      """
      recommendations = []
      # Get the list of indices where the movie is rated and unrated
      unrated_index = np.where(user_ratings == 0)[0]
      rated_index = np.where(user_ratings != 0)[0]

      # For each movie index of an unrated movie
      for i in unrated_index:
          unrated_vec = ratings_matrix[i]
          score = 0

          # For each movie index rated by the user
          for j in rated_index:
              rated_rating = user_ratings[j]
              rated_vec = ratings_matrix[j]
              score += rated_rating * self.similarity(unrated_vec, rated_vec)

          # Add score to the recommendations as a tuple (score of unrated movie, unrated movie index)
          recommendations.append((score, i))

      # Sort the list in descending order and take top k
      recommendations = sorted(recommendations)[::-1][:k]
      # Extract and keep only the movie index
      recommendations = [elem[1] for elem in recommendations]

      return recommendations


    #############################################################################
    # 4. Debug info                                                             #
    #############################################################################

    def debug(self, line):
      """Return debug information as a string for the line string from the REPL"""
      # Pass the debug information that you may think is important for your
      # evaluators
      debug_info = 'debug info'
      return debug_info


    #############################################################################
    # 5. Write a description for your chatbot here!                             #
    #############################################################################
    def intro(self):
      """Return a string to use as your chatbot's description for the user.

      Consider adding to this description any information about what your chatbot
      can do and how the user can interact with it.
      """
      return """
      Your task is to implement the chatbot as detailed in the PA6 instructions.
      Remember: in the starter mode, movie names will come in quotation marks and
      expressions of sentiment will be simple!
      Write here the description for your own chatbot!
      """


if __name__ == '__main__':
  print('To run your chatbot in an interactive loop from the command line, run:')
  print('    python3 repl.py')
