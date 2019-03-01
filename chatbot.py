# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################

# Creative feature flags
# Set to true to enable to false to disable the corresponding feature
use_quoteless_caseless_extraction = True
disambiguate_extracted_titles = True
use_title_spell_correction = False
use_multiple_movies_sentiment_extraction = False
use_arbitrary_input_response = False
use_understand_previous_references = False

import movielens

import numpy as np
import re
import random

from deps import lib

# Edit distance allowed for a movie title match
EDIT_DIST = 3

# Number of recommendations Marvin will suggest
NUM_REC = 10

# Corpuses for different responses
# Opening greeting
greeting_corp = [
"Please tell me about movies you've liked or didn't like so I can make you some recommendations.",
"Please tell me about movies you've watched."
]

goodbye_corp = [
"Thanks for chatting with me! Hope you have a nice day!",
"Hope I was able to recommend some good movies! Bye!",
"Come back for more movie recommendations anytime! Goodbye!"
]
# Movie title successfully extracted but is invalid
invalid_movie_corp = [
"Humm... Sorry I don't think I know about this movie that you mentioned. Please try another one."
]

multi_movie_corp = [
"Ahh, I have found more than one movie called \"{}\". There is {}.\n"
]

spell_corrected_corp_single = [
"I'm sorry, I couldn't find a movie titled \"{}\". Did you happen to mean {}?"
]

spell_corrected_corp_single_error = [
"Sorry, I couldn't understand your answer. Please answer with either \'yes\' or \'no\': {}"
]

spell_corrected_corp_single_no = [
"OK, so you weren't talking about {} after all. \
In that case, please check your spelling and repeat your preference, or try talking about another movie!"
]

spell_corrected_corp = [
"I'm sorry, I couldn't find the movies titled \"{}\". Did you happen to mean {}?\n"
]

pos_movie_corp = [
"You liked {}, got it!",
"Ok, so you enjoyed {}.",
"Oh I remember liking {} too. Great minds think alike huh?",
"You liked {}? Good choice!"
]

neutral_movie_corp = [
"I'm not sure if you liked {} or not. Can you tell me more? Your next response can help me decide.\nYou can also choose to not answer by saying No. "
]

neg_movie_corp = [
"I see. You didn't like {}.",
"{} is no good, noted.",
"Yea I can't say I liked {} very much either."
]

catchall_corp = [
"I see.",
"Interesting.",
"Wow.",
"Sure.",
"Okay.",
"That's good to know.",
"Is that so?",
"Got it."
]

# The prompt above needs to change depending on disambiguation
if disambiguate_extracted_titles:
    multi_movie_corp = [line + "Can you clarify which one you are refering to? You don't have to though. Just say No."\
            for line in multi_movie_corp]
else:
    multi_movie_corp = [line + "Please repeat your preference with a more specific title."\
            for line in multi_movie_corp]

if disambiguate_extracted_titles:
    spell_corrected_corp = [line + "It would be great if you can clarify. Or you can just answer no and we can move on to talking about other movies!"\
            for line in spell_corrected_corp]
else:
    spell_corrected_corp = [line + "Please check your spelling and repeat your preference, or try talking about another movie!"\
            for line in spell_corrected_corp]


class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
      # The chatbot's default name is `moviebot`. Give your chatbot a new name.
      self.name = 'Marvin the Marvelous Moviebot'

      self.creative = creative

      # This matrix has the following shape: num_movies x num_users
      # The values stored in each row i and column j is the rating for
      # movie i by user j
      self.titles, ratings = movielens.ratings()
      self.titles = lib.standardize_titles(self.titles)

      self.sentiment = movielens.sentiment()
      self.sentiment = lib.stem_map(self.sentiment)

      # Binarize the movie ratings before storing the binarized matrix.
      self.ratings = self.binarize(ratings)

      # Vector that keeps track of user movie preference
      self.user_ratings = np.zeros(len(self.titles))

      # Flag that user can update by talking to the chatbot. Set to false when the user asks for a recommendation
      # self.add_review = True

      # Becomes true once the user's made 5 recommendations.
      self.can_recommend = False

      # Flag for if quoteless movie title extraction was performed
      self.quoteless_title_extraction = False

      # Used when Marvin asks if the user wants to accept the spell correction or not
      # Variables for remembering info about the movie that's being spell corrected.
      self.spell_correction_answer = False
      self.spell_correction_prompt = ''
      self.spell_correction_movie_index = 0
      self.spell_correction_movie_title = ''
      self.spell_correction_review = ''

      # Used when Marvin cannot determine if a review is pos or neg and asks for more information
      # Variables for remembering info about the movie that's going to have its preference updated
      self.preference_update_answer = False
      self.preference_update_movie_index = 0
      self.preference_update_movie_title = 0

      # Used when Marvin finds multiple movies that match the description and needs to disambiguate
      # Variables for remembering info about the movies that need clarification
      self.clarify_answer = False
      self.clarify_movie_indicies = []
      self.clarify_review = ''

      # Used to keep track of recommendations
      self.rec_answer = False
      self.recommendations = []
      self.rec_index = 0

      # Used to keep track of the sentiment classification of the last review
      self.last_senti = 0

    #############################################################################
    # 1. WARM UP REPL                                                           #
    #############################################################################

    def greeting(self):
      """Return a message that the chatbot uses to greet the user."""
      greeting_message = lib.getResponse(greeting_corp)

      return greeting_message

    def goodbye(self):
      """Return a message that the chatbot uses to bid farewell to the user."""
      goodbye_message = lib.getResponse(goodbye_corp)

      return goodbye_message


    ###############################################################################
    # 2. Modules 2 and 3: extraction and transformation                           #
    ###############################################################################

    def process(self, line):
      """Process a line of input from the REPL and generate a response.

      This is the method that is called by the REPL loop directly with user input.

      Takes the input string from the REPL and call delegated functions that
        1) extract the relevant information, and
        2) transform the information into a response to the user.

      Example:
        resp = chatbot.process('I loved "The Notebok" so much!!')
        print(resp) // prints 'So you loved "The Notebook", huh?'

      :param line: a user-supplied line of text
      :returns: a string containing the chatbot's response to the user input
      """
      # Handle spell correction response first.
      if self.creative and self.spell_correction_answer:
        self.spell_correction_answer = False
        response = self.process_spell_correction_response(line)

      # Handle user response to update movie preference first.
      # User is allowed to answer No and this part of the code will not execute
      elif self.creative and self.preference_update_answer:
          if line[:2].lower() != 'no':
              senti = self.extract_sentiment(line)
              response = self.process_movie_preference(self.preference_update_movie_index, self.preference_update_movie_title, review=None, usr_senti=senti)
          else:
              response = "Ok, that's fine. Let's move on. Tell me something else."
          self.preference_update_answer = False

      # Handle user response to disambiguate which movie the user is refering to
      # User is allowed to answer No and this part will not execute
      elif self.creative and self.clarify_answer and disambiguate_extracted_titles:
          if line[:2].lower() != 'no':
              # Disambiguate if the user didn't say no
              self.clarify_movie_indices = self.disambiguate(line, self.clarify_movie_indices)
              movies = lib.extract_movies_using_indices(self.titles, self.clarify_movie_indices)

              # If we successfully identified one movie, add its review, otherwise keep asking for clarifications.
              if len(self.clarify_movie_indices) == 1:
                  response = self.process_movie_preference(self.clarify_movie_indices[0], movies[0], self.clarify_review)
                  self.clarify_answer = False
              else:
                  response = 'Thank you for clarifying! However, I still need to decide between {}. \nHelp me out? It\'s ok to say no.'.format(', '.join(movies))
          else:
              response = "Alrighty. Forget about that movie. Tell me another one."
              self.clarify_answer = False

      # Hanldes when the user wants another recommendation
      elif self.rec_answer:
        if 'yes' in line.lower():
            if self.rec_index >= NUM_REC:
                response = "Welp, I already gave you {} recommendations and now I'm out. Tell me more about movies so I can continue to recommend.".format(NUM_REC)
                self.rec_answer = False
            else:
                response = "May I recommend {}.".format(self.recommendations[self.rec_index]) + "\nDo you want another recommendation?"
                self.rec_index += 1
        elif 'no' in line.lower():
            response = "Ok, moving on. Feel free to add more reviews so I can make better recommendations."
            self.rec_answer = False
        else:
            response = "Humm...so is that yes or no?"

      # Check if a user wants a recommendation:
      elif self.can_recommend and "recommend" in line.lower():
        # Recommend movie(s).
        response = "I have found a recommendation for you: \n"
        rec_indices = self.recommend(self.user_ratings, self.ratings, k=NUM_REC, creative=self.creative)
        recs = lib.extract_movies_using_indices(self.titles, rec_indices)
        self.recommendations = recs
        self.rec_index = 1
        self.rec_answer = True
        response += recs[0] + '\nIf you would like to hear another recommendation, say yes, otherwise say no.'

      elif "recommend" in line.lower():
        # Not enough information to recommend a movie.
        return "You need to rate at least 5 movies before I can recommend anything. So what did you like or didn't like?"

      else:
        response = self.add_movie_ratings(line)

      # Check if a user can have a recommendation
      if np.count_nonzero(self.user_ratings) >= 5 and not self.can_recommend:
          self.can_recommend = True
          return response + "\nGreat! Now I have enough information to make recommendations.\n You can continue to rate movies or ask for a recommendation."

      return response

    def process_spell_correction_response(self, line):
      """
      Handles the user's response to a movie-title spelling correction confirmation.
      """
      # Check if responding to a previous spell correction question.
      has_y = 'y' in line.lower()
      has_n = 'n' in line.lower()

      if has_y and not has_n:
        # Yes
        return self.process_movie_preference (
          self.spell_correction_movie_index,
          self.spell_correction_movie_title,
          self.spell_correction_review
        )
      elif has_n and not has_y:
        return lib.getResponse(
          spell_corrected_corp_single_no
          ).format(self.spell_correction_movie_title)
        # No
      else:
        # Unknown
        self.spell_correction_answer = True
        return lib.getResponse(
          spell_corrected_corp_single_error
          ).format(self.spell_correction_prompt)


    def add_movie_ratings(self, line):
      """
      Takes in a user input in the form of a movie review.
      Returns a response in the form of a string. The response either reprompts the user or
      confirms the review is received.
      (starter mode) Extracts only one movie.
      """
      # Extract titles from user input.
      titles = self.extract_titles(line)

      ### No titles are extracted ###
      if len(titles) == 0:
        if use_arbitrary_input_response and self.creative:
          # parse input and see if we can generate some arbitrary response
          return self.generate_arbitrary_response(line)
        else:
          return "I didn't catch that. Did you talk about exactly one movie? Remember to put the movie title in quotes."

      ### More than one title is extracted ###
      elif len(titles) > 1:
        if self.quoteless_title_extraction:
            return ("Sorry, I didn't quite catch that. " +
                "I can only process a single quoteless title currently, and I think you might've mentioned " +
                "{}.\nPlease try encolosing your movie".format(lib.concatenate_titles(
                  [('\"{}\"'.format(t)) for t in titles], 'and')
                ) +
                " titles with \"\" or talking only about a single movie.")
        elif use_multiple_movies_sentiment_extraction and self.creative:
            return self.process_multi_titles(line)
        else:
            return "I didn't catch that. Did you talk about exactly one movie? Remember to put the movie title in quotes."

      ### Exactly one title extracted ###
      else:
        title = titles[0]
        return self.process_single_title(title, line)

    def generate_arbitrary_response(self, line):
      """
      Generates some arbitrary response depending on user input
      """
      if len (re.split(r'\? |! |\. ', line)) > 1:
        # Too many sentences
        return "Woah there, slow down! I can only understand one sentence at a time."

      end_punc = {'.', '!', '?', ',', ';'}
      if line[-1] in end_punc:
        line = line[:-1]

      q_words = {'who', 'what', 'when', 'where', 'why', 'how'}
      yn_q_words = {'did', 'do', 'can', 'may', 'will'}
      tobe_verbs = {'is', 'are', 'was', 'were'}

      i_you_dict = {'me':'you', 'i':'you', 'my':'your', 'myself':'yourself', 'you':'me', 'your':'my', 'yourself':'myself'}

      tokens = line.split()
      if len(tokens) == 0:
        return "Oops, looks like you pressed enter without typing anything :)"
      if len(tokens) <= 2:
        first_word = tokens[0].lower()
        if first_word in q_words or first_word in yn_q_words:
          # echo question
          if len(tokens) == 1:
            return "I don't know - {}?".format(first_word)
          else:
            second_word = tokens[1].lower()
            if second_word in i_you_dict:
              second_word = i_you_dict[second_word]
            return "I don't know - {} {}?".format(first_word, second_word)
        # arbitrary response
        return ("{} But why don't we try talking more about movies?".format(lib.getResponse(catchall_corp))
        + " After all, I am Marvin the Marvelous 'Movie' bot :)")

      first_word = tokens[0]
      second_word = tokens[1]
      last_words = ' '.join(tokens[2:])

      last_words_flipped_builder = []
      for tkn in tokens[2:]:
        if tkn.lower() in i_you_dict:
          last_words_flipped_builder.append(i_you_dict[tkn.lower()])
        else:
          last_words_flipped_builder.append(tkn)
      last_words_flipped = ' '.join(last_words_flipped_builder)

      if first_word.lower() in q_words and second_word.lower() in tobe_verbs:
        return "I don't know {} {} {}.".format(first_word.lower(), last_words_flipped, second_word.lower())

      elif first_word.lower() == 'can' and second_word.lower() == 'you':
        return "Sorry, but I probably can't {}.".format(last_words_flipped)

      elif first_word.lower() == 'did' and second_word.lower() == 'you':
        return "I'm not sure, but I probably didn't {}.".format(last_words_flipped)

      elif first_word.lower() == 'do' and second_word.lower() == 'you':
        return "Sorry, but I probably don't {}.".format(last_words_flipped)

      elif first_word.lower() == 'will' and second_word.lower() == 'you':
        return "I'm not sure, but I probably won't {}.".format(last_words_flipped)

      elif first_word.lower() in q_words or first_word.lower() in yn_q_words:
        return "I don't know, {}? The world is full of mysteries...".format(first_word.lower() + ' ' + second_word + ' ' + last_words)

      elif 'thank' in line.lower():
        return "You're welcome!"

      else:
        return ("{} But why don't we try talking more about movies?".format(lib.getResponse(catchall_corp))
        + " After all, I am Marvin the Marvelous 'Movie' bot :)")

    def process_multi_titles(self, line):
      """
      Used by add_movie_ratings when many movies titles are found
      Handles the case where the user supplies multiple movie titles.
      """
      movie_sentiments = self.extract_sentiment_for_movies(line)
      pos_titles = []
      neg_titles = []
      neutral_titles = []
      unprocessable_titles = []

      for elem in movie_sentiments:
        movie_title, sentiment = elem
        movie_indexes = self.find_movies_by_title(movie_title)
        title = '\"' + movie_title + '\"'

        if len(movie_indexes) == 1:
          self.process_movie_preference(movie_indexes[0], movie_title, None, sentiment)

          if sentiment < 0:
            neg_titles.append(title)
          elif sentiment == 0:
            # TODO Currently does not support multiple movies preference update(if one is neutral)
            self.preference_update_answer = False
            neutral_titles.append(title)
          else:
            pos_titles.append(title)
        else:
          unprocessable_titles.append(title)

      # Build response.
      response = ''
      prev_sentence = False

      if len(pos_titles) > 0:
        response += "You liked {}.".format(lib.concatenate_titles(pos_titles, 'and'))
        prev_sentence = True
      if len(neg_titles) > 0:
        if prev_sentence:
          response += '\n'
        response += "You didn't like {}.".format(lib.concatenate_titles(neg_titles, 'and'))
        prev_sentence = True
      if len(neutral_titles) > 0:
        if prev_sentence:
          response += '\n'
        response += "I couldn't really tell if you liked {}.".format(lib.concatenate_titles(neutral_titles, 'and'))
        prev_sentence = True
      if len(unprocessable_titles) > 0:
        if prev_sentence:
          response += '\n'
        response += ("I couldn't locate a single specific movie for"
          + " {} ".format(lib.concatenate_titles(unprocessable_titles, 'or'))
          + "- Please check your spelling or try specifying the year,"
          + " e.g. \"Titanic (1997)\".")

      return response

    def process_single_title(self, title, line):
      """
      Used by add_movie_ratings when exactly one movie title is found
      Handles the case where the user supplies multiple movie titles.
      """
      # Search for a matching movie.
      movie_index = self.find_movies_by_title(title)
      spell_corrected = False

      # Try enabling spell correction if no movies were found.
      if use_title_spell_correction and self.creative and len (movie_index) == 0:
        movie_index = self.find_movies_closest_to_title(title)
        spell_corrected = True

      movies = [('\"' + m + '\"') for m in lib.extract_movies_using_indices(self.titles, movie_index)]

      # No movies found
      if len(movies) == 0:
        return lib.getResponse(invalid_movie_corp)

      # One movie is found with spell correction
      elif len(movies) == 1 and spell_corrected:
          self.spell_correction_answer = True
          self.spell_correction_prompt = lib.getResponse(
            spell_corrected_corp_single).format(title, movies[0])
          self.spell_correction_movie_index = movie_index[0]
          self.spell_correction_movie_title = movies[0]
          self.spell_correction_review = line

          return self.spell_correction_prompt

      # More than one movie is found
      elif len(movies) > 1:

        if disambiguate_extracted_titles:
          # Prepare for clarifications from user
          self.clarify_answer = True
          self.clarify_movie_indices = movie_index
          self.clarify_review = line

        # Build movies list with correct grammar. Final conjunction depends on case.
        if spell_corrected:
          formatted_movies = lib.concatenate_titles(movies, 'or')
          return lib.getResponse(spell_corrected_corp).format(title, formatted_movies)
        else:
          formatted_movies = lib.concatenate_titles(movies, 'and')
          return lib.getResponse(multi_movie_corp).format(title, formatted_movies)

      # Exactly one movie is found
      return self.process_movie_preference(movie_index[0], movies[0], line)


    def process_movie_preference (self, movie_index, movie_title, review, usr_senti=None):
      """ Performs sentiment extraction on the user's review and updates the
      user_rating for the specified movie. Returns the bot's response to the
      user as implicit confirmation.
      """
      if usr_senti is None:
        sentiment = self.extract_sentiment(review)
      else:
        sentiment = usr_senti

      # Provide ackowledgement
      if sentiment == 1:
        self.user_ratings[movie_index] = 1
        return lib.getResponse(pos_movie_corp).format(movie_title)

      elif sentiment == 0:
        self.preference_update_answer = True
        self.preference_update_movie_title = movie_title
        self.preference_update_movie_index = movie_index
        return lib.getResponse(neutral_movie_corp).format(movie_title)

      else:
        self.user_ratings[movie_index] = -1
        return lib.getResponse(neg_movie_corp).format(movie_title)


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

      quote_pat = '\"(?P<title>[^\"]+)\"'
      e_matches = re.findall(quote_pat, text)
      for title in e_matches:
        titles.append(title)

      self.quoteless_title_extraction = False
      if use_quoteless_caseless_extraction and self.creative and len(titles) == 0:
        # Attempt to extract titles without explicit quotation marks
        for elem in self.titles:
          title, year, _ = elem
          title_with_year = title + ' ({})'.format(year)

          if lib.extract_title_by_word(title_with_year.lower(), text.lower()):
            # Title and year together are unique identifiers of a movie
            return [title_with_year,]
          elif lib.extract_title_by_word(title.lower(), text.lower()):
            # Remove existing titles that are substrings of current title
            titles_to_remove = []
            for t in titles:
              if t.lower() in title.lower():
                titles_to_remove.append(t)
            for t in titles_to_remove:
              titles.remove(t)

            # Append current title to list only if not substring of any existing title
            append_title = True
            for t in titles:
              if title.lower() in t.lower():
                append_title = False
            if append_title:
              titles.append(title)

        if len(titles) > 0:
          self.quoteless_title_extraction = True

      return titles

    def find_movies_by_title(self, title, max_distance=-1):
      """ Given a movie title, return a list of indices of matching movies.

      - If no movies are found that match the given title, return an empty list.
      - If multiple movies are found that match the given title, return a list
      containing all of the indices of these matching movies.
      - If exactly one movie is found that matches the given title, return a list
      that contains the index of that matching movie.

      If max_distance is provided, then enable max-distance spell correcting.
      Note that the spell correcting is applied after year extraction.

      Example:
        ids = chatbot.find_movies_by_title('Titanic')
        print(ids) // prints [1359, 1953]

      :param title: a string containing a movie title
      :returns: a list of indices of matching movies
      """
      movies = []

      movie_title = title
      movie_year = None

      # Minimun edit distance found for all titles
      min_dist = max_distance

      # Extract year from title
      movie_title, movie_year = lib.extract_year_from_title(title)

      # Standardize leading article position (move to front)
      movie_title = lib.move_leading_article_to_front(movie_title)

      # Search through movie title database for matching titles
      for i in range(len(self.titles)):
        entry_title, entry_year, genre = self.titles[i]

        # Filter by movie year
        if movie_year is not None and movie_year != entry_year:
          continue

        # Filter by movie title
        if max_distance <= 0:
            # No spelling correction
            if self.creative:
              # Handle differing capitalizations
              # If user inputs "scream", should find all movies containing "scream" but not "screams" / "screaming"
              # Only if movie was found using quotes, otherwise just use lower case comparison to reduce overflowing matches
              if lib.title_contains_words(movie_title, entry_title) and not self.quoteless_title_extraction:
                  movies.append(i)
              elif movie_title.lower() == entry_title.lower():
                  movies.append(i)
            else:
              if movie_title == entry_title:
                  movies.append(i)
        else:
            dist = lib.min_edit_distance(movie_title.lower(), entry_title.lower())
            if dist <= max_distance:
                movies.append((dist, i))
                min_dist = min(min_dist, dist)

      # Only keep movies that tie for smallest edit distance
      if not max_distance <= 0:
          min_movies = []
          for movie in movies:
              if movie[0] == min_dist:
                  min_movies.append(movie[1])
          movies = min_movies

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
      negations = ['don\'t', 'not', 'never', 'none', 'nothing', 'hardly', 'didn\'t', 'no']
      negation_scale = 1

      resets = ['but', 'however']

      # Remove movie names from the text
      text = text.lower()
      titles = self.extract_titles(text)
      for title in titles:
        text = text.replace(title.lower(), '')

      words_stemmed = lib.stem_text(text)
      words = words_stemmed.split()
      posCount = 0
      #print(words)
      for word in words:
          # All words after the negation will take on its inverted value
          if word in negations:
              negation_scale *= -1
          # Whatever comes before reset words don't really matter
          elif word in resets:
              posCount = 0

          # Reset the negation scale to 1 if at end punctuation. Commas don't count.
          if not word[len(word) - 1].isalpha() and word[len(word) - 1] != ",":
              negation_scale = 1

          if word in self.sentiment:
              if self.sentiment[word] == 'pos':
                  #print("Word: " + word + " score: " + str(negation_scale))
                  posCount += negation_scale
              else:
                  #print("Word: " + word + " score: " + str(-negation_scale))
                  posCount -= negation_scale

      if posCount != 0:
          posCount = posCount / abs(posCount)
          # Only update last_senti if a another sentiment word is extracted
          self.last_senti = posCount

      # Try to utilize memory of the last review to make a decision about sentiment
      elif use_understand_previous_references and self.creative and self.last_senti != 0:
          posCount = self.last_senti
          negations.append('but')
          for neg_conj in negations:
              if neg_conj in words:
                  posCount *= -1
                  break

      #print(posCount)
      return posCount

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
      # Enums for tracking token types
      TKN_TITLE = 0
      TKN_CONJ = 1
      TKN_OTHER = 2

      tagged_tokens = []

      # TODO: need to tokenize by non-quote movies as well

      # Split text by sentence, then tokenize by conjuctions, movie titles, and other words
      for sentence in re.split(r'\? |! |\. ', text):
        tks = lib.tokenize_conj_movie_other(sentence)
        tks.append((TKN_OTHER, '.'))
        tagged_tokens = tagged_tokens + tks

      # Extract movie sentiments
      movie_sentiments = []

      current_sentence = ''
      current_movies = []
      prev_neg_sentiment = None

      for elem in tagged_tokens:
        tag, token = elem

        # skip conjunctions and movie titles
        if tag == TKN_OTHER:
          current_sentence += token + ' '

        # append to current_movies
        if tag == TKN_TITLE:
          current_movies.append(token)

        # check if end of sentence / phrase ('but' is the only conj that signals end of phrase)
        if (tag == TKN_CONJ and token.lower() == 'but') or (tag == TKN_OTHER and token == '.'):
          sentiment = self.extract_sentiment(current_sentence)

          # If neither nor exists in the current sentence, its sentiment is inverted.
          # Neither nor and only be checke at the end of sentence because neither usually appears before the movie title
          # "I liked neither this nor that"
          if "neither" in current_sentence.lower():
              sentiment *= -1

          # If current sentence seg has neutral sentiment and previous sentence seg
          # ended with a negation conjunction ('but')
          if (sentiment == 0) and (prev_neg_sentiment is not None):
            sentiment = prev_neg_sentiment * -1

          for movie in current_movies:
            movie_sentiments.append((movie, sentiment))

          # Set / reset variables and continue processing
          if (tag == TKN_CONJ and token.lower() == 'but'):
            # Track previous sentance clause sentiment.
            prev_neg_sentiment = sentiment
          if (tag == TKN_OTHER and token == '.'):
            # Reset negated sentiment tracker.
            prev_neg_sentiment = None

          current_sentence = ''
          current_movies = []

      return movie_sentiments

    def find_movies_closest_to_title(self, title, max_distance=EDIT_DIST):
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
      clarification = clarification.lower()

      # List of full movies names with years of all candidates
      movie_list = lib.extract_movies_using_indices(self.titles, candidates)
      # Should be set to True once the list shrinks
      clarified = False

      ### Use direct comparison ###
      if not clarified:
          potential_candidates = [candidates[i] for i in range(len(candidates)) if clarification.strip() == movie_list[i].lower()]
          if len(potential_candidates) < len(candidates) and potential_candidates != []:
              candidates = potential_candidates
              clarified = True

      ### Clarification by year ###
      # Assume the user enters a sentence that contains exactly one year number which clarifies
      year_pat = r'\b\(?([12][0-9]{3})\)?\b'
      year_matches = re.findall(year_pat, clarification)

      if len(year_matches) == 1 and not clarified:
          year = year_matches[0]

          # Loops through each movie and for every movie that contains the year, store that movie's index in new candidates
          # Now candidates should contain movie indices for movies that have the year somewhere in its title
          candidates = [candidates[i] for i in range(len(candidates)) if lib.title_contains_words(year, movie_list[i])]
          clarified = True

      ### Clarification by name ###
      # Assume the user enters an exact substring of the movie title. eg: Using "Scorcerer's Stone" to clarify "Harry Potter and the Scorcerer's Stone"
      if not clarified:
          potential_candidates = [candidates[i] for i in range(len(candidates)) if lib.title_contains_words(clarification.strip(), movie_list[i][:-6])] #-6 because we don't include the year of the movie
          if len(potential_candidates) < len(candidates) and potential_candidates != []:
              candidates = potential_candidates
              clarified = True

      return candidates

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
      norm1 = np.linalg.norm(u)
      norm2 = np.linalg.norm(v)
      if norm1 * norm2 == 0:
          return 0
      return np.dot (u, v) / (norm1 * norm2)


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
      Hello there! I'm Marvin the Marvelous Moviebot!

      (∩｀-´)⊃━☆ﾟ.*･｡ﾟ

      I can recommend movies to you based on your preferences. You simply need
      to tell me about some of the movies you've watched before and what you
      thought of each movie. After I have gathered enough information, I'll let
      you know that I'm ready to make a recommendation. You can then request a
      recommendation, and I'll work my magic!

      ¸¸.•*¨*•♫♪¸¸.•*¨*•♫♪¸¸.•*¨*•♫♪¸¸.•*¨*•♫♪¸¸.•*¨*•♫♪¸¸.•*¨*•♫♪¸¸.•*¨*•♫♪

      Creative features:

      To enable each creative feature while chatting with me, set its
      corresponding global flag in \"chatbot.py\" (line 6) to True. Likewise,
      to disable a creative feature, set its corresponding flag to False.
      Note that the '--creative' flag must also be specified in order for any
      creative feature to take effect.

      The implemented creative features (and their associated flags) are:

      - Identifying movies without quotation marks and correct capitalization
        (part 1)
        Flag: use_quoteless_caseless_extraction

      - Identifying movies without quotation marks and correct capitalization
        (part 2)
        Flag: <NO FLAG>, specifiying the '--creative' flag enables this feature

      - Alternate/foreign titles
        Flag: <NO FLAG>, specifiying the '--creative' flag enables this feature

      - Disambiguation (part 1)
        Flag: disambiguate_extracted_titles

      - Understanding references to things said previously
        Flag: use_understand_previous_references
        E.g. User: 'I saw "Avatar"', Marvin: 'What did you think of "Avatar"?',
        User: 'I liked it.', Marvin: 'You liked "Avatar"'.
        E.g. User: 'I liked "Avatar"', Marvin: 'You liked "Avatar!"', User: 'But
        not "I, robot".', Marvin: 'I see, you didn't like "I, Robot".'

      - Responding to arbitrary input
        Flag: use_arbitrary_input_response

      - Dialogue for spell-checking
        Flag: use_title_spell_correction

      - Dialogue for disambiguation
        Flag: disambiguate_extracted_titles
        E.g. User: 'I liked "Harry Potter"', Marvin: 'Which "Harry Potter" are
        you refering to?', User: 'Chamber of Secrets',
        Marvin: 'You liked Harry Potter and the Chamber of Secrets (2002)'

      - Communicating sentiments and movies extracted to the user given
        multiple-movie input
        Flag: use_multiple_movies_sentiment_extraction

      - Speaking very fluently
        Am I your best friend now? \\(^-^)/
      """


if __name__ == '__main__':
  print('To run your chatbot in an interactive loop from the command line, run:')
  print('    python3 repl.py')
