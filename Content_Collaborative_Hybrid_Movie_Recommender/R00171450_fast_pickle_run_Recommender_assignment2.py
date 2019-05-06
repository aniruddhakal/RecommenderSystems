import sys
from random import random
from time import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.spatial.distance import jaccard
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

class Collaborative_Filtering_Recommender:

    def __init__(self, user_ratings_file, save_pickle_flag=False, read_from_pickle_flag=False, pickle_dir=""):

        self.user_ratings = None
        self.movie_ratings = None
        # self.movies_dict = {}
        # self.ratings = None
        self.user_pearson_similarities = None
        self.movie_pearson_similarities = None
        self.movie_cosine_similarities = None
        self.user_cosine_similarities = None
        self.movie_adjusted_cosine_similarities = None
        self.user_adjusted_cosine_similarities = None
        self.movie_mean_centers = None
        self.user_mean_centers = None
        self.users = []
        self.movies = []
        self.user_ratings_file = user_ratings_file
        self.read_from_pickle = read_from_pickle_flag
        self.pickle_dir = pickle_dir
        self.save_pickle = save_pickle_flag

    def load_ratings(self):
        rows_n_columns = pd.read_csv(filepath_or_buffer=self.user_ratings_file, sep="\s+", usecols=range(3),
                                     names=["userid", "movieid", "rating"])

        self.user_ratings = pd.pivot_table(rows_n_columns, values="rating", index="userid", columns="movieid")

        # self.movie_ratings = pd.pivot_table(rows_n_columns, values="rating", index="movieid", columns="userid")
        self.movie_ratings = self.user_ratings.transpose(copy=True)

        self.update_users_and_movies_lists()

    def get_common_ratings(self, df, user1_id=None, user2_id=None, movie1_id=None, movie2_id=None):
        if user1_id is not None and user2_id is not None:
            x_ratings = df.loc[user1_id, :]
            y_ratings = df.loc[user2_id, :]
        elif movie1_id is not None and movie2_id is not None:
            x_ratings = df.loc[movie1_id, :]
            y_ratings = df.loc[movie2_id, :]
        else:
            raise ValueError(
                "get_common_ratings: please provide either 2 valid user id's or movie id's")

        array1 = np.logical_not(np.isnan(x_ratings.tolist()))
        array2 = np.logical_not(np.isnan(y_ratings.tolist()))

        common_ratings_indices = np.logical_and(array1, array2)

        x = x_ratings.loc[common_ratings_indices].values
        y = y_ratings.loc[common_ratings_indices].values

        return x, y

    def compute_user_similarities(self, similarity_function):
        target_df, source_df = self.compute_dependencies_and_get_target_df(similarity_function, "user")

        if target_df is None:
            print("Computing user similarities for %s" % str(similarity_function))
            user_sim = {}

            for user in self.users:
                for other_user in self.users:
                    # condition to avoid same user and double computations
                    # time complexity: n/2
                    if user != other_user and other_user > user:
                        x, y = self.get_common_ratings(source_df, user1_id=user, user2_id=other_user)
                        similarity = similarity_function(x, y)

                        if user not in user_sim:
                            user_sim[user] = {}

                        if other_user not in user_sim:
                            user_sim[other_user] = {}

                        user_sim[user][other_user] = similarity
                        user_sim[other_user][user] = similarity

            target_df = pd.DataFrame(user_sim)
            self.update_target_df(similarity_function, target_df, "user")
            print("Finished Computing user similarities for %s" % str(similarity_function))

    def update_target_df(self, similarity_function, target_df, string):
        if string == "movie":
            if similarity_function == pearson_correlation:
                self.movie_pearson_similarities = target_df
                self.save_pickle and pd.to_pickle(self.movie_pearson_similarities,
                                                  "./input/pickle/collaborative/movie_pearson_similarities",
                                                  protocol=4, compression='bz2')
            elif similarity_function == cosine_similarity:
                self.movie_cosine_similarities = target_df
            elif similarity_function == adjusted_cosine_similarity:
                self.movie_adjusted_cosine_similarities = target_df
                self.save_pickle and pd.to_pickle(self.movie_adjusted_cosine_similarities,
                                                  "./input/pickle/collaborative/movie_adjusted_cosine_similarities",
                                                  protocol=4, compression='bz2')
        elif string == "user":
            if similarity_function == pearson_correlation:
                self.user_pearson_similarities = target_df
                self.save_pickle and pd.to_pickle(self.user_pearson_similarities,
                                                  "./input/pickle/collaborative/user_pearson_similarities",
                                                  protocol=4, compression='bz2')
            elif similarity_function == cosine_similarity:
                self.user_cosine_similarities = target_df
            elif similarity_function == adjusted_cosine_similarity:
                self.user_adjusted_cosine_similarities = target_df
                self.save_pickle and pd.to_pickle(self.user_adjusted_cosine_similarities,
                                                  "./input/pickle/collaborative/user_adjusted_cosine_similarities",
                                                  protocol=4, compression='bz2')

    def compute_dependencies_and_get_target_df(self, similarity_function, string):
        target_df = None
        source_df = None

        if string == "movie":
            source_df = self.movie_ratings

            if similarity_function == pearson_correlation:
                target_df = self.movie_pearson_similarities
            elif similarity_function == cosine_similarity:
                target_df = self.movie_cosine_similarities
            elif similarity_function == adjusted_cosine_similarity:
                self.compute_mean_centering()
                target_df = self.movie_adjusted_cosine_similarities
                source_df = self.user_mean_centers
        elif string == "user":
            source_df = self.user_ratings

            if similarity_function == pearson_correlation:
                target_df = self.user_pearson_similarities
            elif similarity_function == cosine_similarity:
                target_df = self.user_cosine_similarities
            elif similarity_function == adjusted_cosine_similarity:
                self.compute_mean_centering()
                target_df = self.user_adjusted_cosine_similarities
                source_df = self.user_mean_centers

        return target_df, source_df

    def compute_movies_similarities(self, similarity_function):
        target_dataframe, source_dataframe = self.compute_dependencies_and_get_target_df(similarity_function, "movie")

        if target_dataframe is None:
            print("Computing movie similarities for %s" % str(similarity_function))
            movies_sim = {}

            for movie in self.movies:
                for other_movie in self.movies:
                    # condition to avoid same movie and double computations
                    if movie != other_movie and other_movie > movie:
                        x, y = self.get_common_ratings(source_dataframe, movie1_id=movie, movie2_id=other_movie)
                        # TODO replace with appropriate similarity function call
                        similarity = similarity_function(x, y)

                        if movie not in movies_sim:
                            movies_sim[movie] = {}

                        if other_movie not in movies_sim:
                            movies_sim[other_movie] = {}

                        movies_sim[movie][other_movie] = similarity
                        movies_sim[other_movie][movie] = similarity

            target_dataframe = pd.DataFrame(movies_sim)
            self.update_target_df(similarity_function, target_dataframe, "movie")
            print("Finished Computing movie similarities for %s" % str(similarity_function))

    def update_users_and_movies_lists(self):
        dataframe = self.user_ratings

        self.users = np.sort(dataframe.index.values)
        self.movies = np.sort(dataframe.columns.values)

    def get_all_nearest_users(self, similarity_function, active_user_id, candidate_movie_id=None):
        if not self.read_from_pickle:
            self.compute_user_similarities(similarity_function)
        else:
            self.user_pearson_similarities = pd.read_pickle(self.pickle_dir + "user_pearson_similarities", compression='bz2')

        target_df, source_df = self.compute_dependencies_and_get_target_df(similarity_function, "user")

        x_list = target_df.loc[active_user_id, :]
        # user_similarities
        ratings = np.logical_not(np.isnan(x_list.tolist()))
        all_users_id = x_list[ratings].index.values

        if candidate_movie_id is not None:
            # filter users who rated candidate movie id
            for index, user in enumerate(all_users_id.tolist()):
                # if np.isnan(self.ratings.loc[candidate_movie_id, user]):
                if np.isnan(self.user_ratings.loc[user, candidate_movie_id]):
                    ratings[index] = False

        all_users_id = x_list[ratings].index.values
        all_ratings = x_list[ratings].values
        sorted_indices = np.argsort(-all_ratings)

        return all_users_id[sorted_indices], all_ratings[sorted_indices]

    def get_all_nearest_movies(self, similarity_function, candidate_movie_id, active_user_id=None):
        if not self.read_from_pickle:
            self.compute_movies_similarities(similarity_function)
        else:
            # TODO enable to compute adjusted cosine - disabled because it takes long time and pickle file isn't saved
            # self.movie_adjusted_cosine_similarities = pd.read_pickle(
            #     self.pickle_dir + "movie_adjusted_cosine_similarities", compression='bz2')
            self.movie_pearson_similarities = pd.read_pickle(self.pickle_dir + "movie_pearson_similarities", compression='bz2')

        target_df, source_df = self.compute_dependencies_and_get_target_df(similarity_function, "movie")

        similarities = target_df.loc[:, candidate_movie_id]
        # movie similarities
        non_na_similarities = np.logical_not(np.isnan(similarities.tolist()))
        all_movies_id = similarities[non_na_similarities].index.values

        if active_user_id is not None:
            for index, movie in enumerate(all_movies_id.tolist()):
                if np.isnan(self.movie_ratings.loc[candidate_movie_id, active_user_id]):
                    non_na_similarities[index] = False

        all_movies_id = similarities[non_na_similarities].index.values
        all_similarities = similarities[non_na_similarities].values
        sorted_indices = np.argsort(-all_similarities)

        return all_movies_id[sorted_indices], all_similarities[sorted_indices]

    def get_k_nearest_users(self, similarity_function, k, active_user_id, candidate_movie_id=None):
        nearest_users, nearest_ratings = self.get_all_nearest_users(similarity_function, active_user_id,
                                                                    candidate_movie_id)

        k_nearest_users = nearest_users[:k]

        return k_nearest_users

    def get_k_thresholded_nearest_movies(self, similarity_function, k, threshold, candidate_movie_id,
                                         active_user_id=None):
        nearest_movies, nearest_similarities = self.get_all_nearest_movies(similarity_function, candidate_movie_id,
                                                                           active_user_id)

        # filter above threshold
        nearest_movies = nearest_movies[nearest_similarities >= threshold]

        k_nearest_movies = nearest_movies[:k]

        return k_nearest_movies

    def get_k_thresholded_nearest_users(self, similarity_function, k, threshold, active_user_id,
                                        candidate_movie_id=None):
        nearest_users, nearest_similarities = self.get_all_nearest_users(similarity_function, active_user_id,
                                                                         candidate_movie_id)

        # filter above threshold
        nearest_users = nearest_users[nearest_similarities >= threshold]

        # filter k-nearest
        k_nearest_users = nearest_users[:k]

        return k_nearest_users

    def get_k_nearest_movies(self, similarity_function, k, candidate_movie_id, active_user_id=None):
        nearest_movies, nearest_similarities = self.get_all_nearest_movies(similarity_function, candidate_movie_id,
                                                                           active_user_id)

        k_nearest_movies = nearest_movies[:k]
        k_nearest_similarities = nearest_similarities[:k]

        return k_nearest_movies, k_nearest_similarities

    def compute_mean_centering(self):
        if self.movie_mean_centers is None:
            print("Computing all Movie Mean Centered Values...")
            self.movie_mean_centers = self.movie_ratings.apply(lambda x: x - np.mean(x), axis=1)

        if self.user_mean_centers is None:
            print("Computing all User Mean Centered Values...")
            self.user_mean_centers = self.user_ratings.apply(lambda x: x - np.mean(x), axis=1).transpose()


class Content_Based_Recommender:

    def __init__(self, movies_file, user_terms_file, user_ratings_file, verbose_flag=False, save_pickle=False,
                 read_from_pickle_flag=False,
                 pickle_dir=""):
        self.movies_file = movies_file
        self.user_terms_file = user_terms_file
        self.user_ratings_file = user_ratings_file
        self.user_terms_df = None
        self.movie_terms_df = None
        self.movie_ratings_df = None
        self.movie_ratings_all_users_avg = None
        self.sorted_all_user_avg_ratings = None
        self.movie_movie_all_similarities_df = None
        self.user_user_all_similarities_df = None
        self.movie_user_all_similarities_df = None
        self.movie_movie_jaccard_similarities_df = None
        self.movie_movie_cosine_similarities_df = None
        self.movie_user_jaccard_similarities_df = None
        self.movie_user_cosine_similarities_df = None
        self.verbose = verbose_flag
        self.read_from_pickle = read_from_pickle_flag
        self.pickle_dir = pickle_dir
        self.save_pickle = save_pickle

    def load_ratings(self):
        self.verbose and print("Loading all ratings")

        start = time()

        # ratings_file = "./input/movies.txt"
        column_names = ["id", "name", "release_date", "url"]
        column_names.extend(list(map(str, list(range(1, 20)))))
        self.movie_terms_df = pd.read_csv(self.movies_file, sep="|", header=None, index_col="id", names=column_names)

        # user_terms_file = "./input/userterms.txt"
        column_names = ['id', 'age', 'gender', 'occupation']
        column_names.extend(list(map(str, list(range(1, 20)))))
        # read userterms.txt
        f = open(self.user_terms_file)
        text = f.read().replace(',', '')
        f.close()

        self.user_terms_df = pd.read_csv(pd.compat.StringIO(text), sep="[\s+\[\]]", header=None, engine='python')
        self.user_terms_df.dropna(axis=1, inplace=True, how='all')
        self.user_terms_df.columns = column_names
        self.user_terms_df.set_index(self.user_terms_df['id'], drop=True, inplace=True)
        self.user_terms_df.drop('id', axis=1, inplace=True)

        # ratings_file = './input/ratings_1.txt'
        rows_n_columns = pd.read_csv(filepath_or_buffer=self.user_ratings_file, sep="\s+", usecols=range(3),
                                     names=["userid", "movieid", "rating"])

        self.movie_ratings_df = pd.pivot_table(rows_n_columns, values="rating", index="userid",
                                               columns="movieid").transpose()

        # Ties in similarity should be broken based on average
        # rating of the item across all users.
        self.movie_ratings_all_users_avg = self.movie_ratings_df.mean(axis=1)
        self.sorted_all_user_avg_ratings = self.movie_ratings_all_users_avg.sort_values(ascending=False).index.tolist()

        finish = time() - start

        self.verbose and print("Finished loading all ratings in %f seconds" % finish)

    def load_from_pickle(self):
        self.verbose and print("Loading pickle movie_movie_all_similarities_df")
        self.movie_movie_all_similarities_df = pd.read_pickle(self.pickle_dir + "movie_movie_all_similarities_df", compression='bz2')

        self.verbose and print("Loading pickle user_user_all_similarities_df")
        self.user_user_all_similarities_df = pd.read_pickle(self.pickle_dir + "user_user_all_similarities_df", compression='bz2')

        self.fast_compute_movie_user_similarities()

    def compute_similarities(self):
        self.verbose and print("Computing all similarities")

        start = time()

        if not self.read_from_pickle:
            self.movie_movie_all_similarities_df = self.fast_compute_entity_entity_similarities(self.movie_terms_df,
                                                                                                self.movie_movie_all_similarities_df)
            self.save_pickle and pd.to_pickle(self.movie_movie_all_similarities_df,
                                              path=self.pickle_dir + "movie_movie_all_similarities_df", protocol=4, compression='bz2')

            self.user_user_all_similarities_df = self.fast_compute_entity_entity_similarities(self.user_terms_df,
                                                                                              self.user_user_all_similarities_df)
            self.save_pickle and pd.to_pickle(self.user_user_all_similarities_df,
                                              path=self.pickle_dir + "user_user_all_similarities_df", protocol=4, compression='bz2')

            self.fast_compute_movie_user_similarities()
        else:
            self.load_from_pickle()

        finish = time() - start

        self.verbose and print("Finished computing all similarities in %f seconds" % finish)

    def fast_compute_entity_entity_similarities(self, source_df, target_df, recompute=False):
        if source_df is None:
            self.load_ratings()
        elif (target_df is not None) and (not recompute):
            return

        self.verbose and print("Computing entity-entity similarities")

        start = time()

        # only fetching the movie terms
        df = source_df.loc[:, '1':'19']

        entity_indices = df.index.values
        entity_indices.sort()

        # compute similarity for movie term x with all other movies in xdf
        def process_df(x, xdf):
            return xdf.apply(lambda y: ((1 - jaccard(x, y)), (1 - cosine(x, y))), axis=1)
            # TODO this may introduce further difficulties, so skip for now
            #   return xdf.apply(lambda y: ((1 - jaccard(x, y)), (1 - cosine(x, y))) if (x.name < y.name) else None, axis=1)
            # TODO try later something like this to fill the other side of matrix, for now, return 1,2 when queried for 2,1
            #   return xdf.apply(lambda y: ((1-jaccard(x, y)), (1-cosine(x, y))) if (x.name < y.name) else (df.loc[y.name, x.name]), axis=1)

        # x on axis=1 is a series of movie_terms
        # with x dropped, we have terms of all other movies in df passed to function process_df()
        target_df = df.apply(lambda x: process_df(x, df.drop(x.name)), axis=1)

        finish = time() - start

        self.verbose and print("Entity-Entity similarities calculation time: %f seconds" % finish)

        return target_df

    def fast_compute_movie_user_similarities(self, recompute=False):
        self.verbose and print("Computing all Movie User similarities")

        start = time()

        if not self.read_from_pickle:
            if (self.user_terms_df is None) or (self.movie_terms_df is None):
                self.load_ratings()
            elif (self.movie_user_all_similarities_df is not None) and (not recompute):
                return

            user_terms_only_df = self.user_terms_df.loc[:, '1':'19']
            movie_terms_only_df = self.movie_terms_df.loc[:, '1':'19']

            d = movie_terms_only_df.apply(
                lambda x: user_terms_only_df.apply(lambda y: (1 - jaccard(x, y), 1 - cosine(x, y)), axis=1), axis=1)

            self.movie_user_all_similarities_df = pd.DataFrame(d)
            self.save_pickle and pd.to_pickle(self.movie_user_all_similarities_df,
                                              path=self.pickle_dir + "movie_user_all_similarities_df", protocol=4, compression='bz2')
        else:
            self.verbose and print("Loading pickle movie_user_all_similarities_df")
            self.movie_user_all_similarities_df = pd.read_pickle(self.pickle_dir + "movie_user_all_similarities_df", compression='bz2')

        finish = time() - start

        self.verbose and print("Finished computing all Movie-User similarities in %f seconds" % finish)

    def compute_entity_entity_similarities(self, source_df):
        start = time()

        entity_indices = source_df.index.values

        entity_indices.sort()

        cosine_similarities = {}
        jaccard_similarities = {}

        for entity1_id in entity_indices:
            other_entities = np.setdiff1d(entity_indices, entity1_id)

            for entity2_id in other_entities:

                # this condition helps avoid redundant calculations
                # e.g. considering similarity for 1,2 is calculated before,
                #       because while calculating sim for 1,2, same values are also stored for 2,1
                #       hence similarity calculation for 2,1 is avoided next time
                if not entity1_id < entity2_id:
                    entity1_terms = source_df.loc[entity1_id, '1':'19'].astype(np.int64)
                    entity2_terms = source_df.loc[entity2_id, '1':'19'].astype(np.int64)

                    jaccard_similarity = 1 - jaccard(entity1_terms, entity2_terms)
                    cosine_similarity = 1 - cosine(entity1_terms, entity2_terms)

                    if entity1_id not in cosine_similarities:
                        cosine_similarities[entity1_id] = {}
                    if entity2_id not in cosine_similarities:
                        cosine_similarities[entity2_id] = {}

                    if entity1_id not in jaccard_similarities:
                        jaccard_similarities[entity1_id] = {}
                    if entity2_id not in jaccard_similarities:
                        jaccard_similarities[entity2_id] = {}

                    # if need to add more similarity calculations, add them here
                    cosine_similarities[entity1_id][entity2_id] = cosine_similarity
                    cosine_similarities[entity2_id][entity1_id] = cosine_similarity

                    jaccard_similarities[entity1_id][entity2_id] = jaccard_similarity
                    jaccard_similarities[entity2_id][entity1_id] = jaccard_similarity

        jaccard_target_df = pd.DataFrame(jaccard_similarities)
        cosine_target_df = pd.DataFrame(cosine_similarities)

        jaccard_target_df.fillna(1, inplace=True)
        cosine_target_df.fillna(1, inplace=True)
        finish = time() - start

        self.verbose and print("Entity-Entity similarities calculation time: %f seconds" % finish)

        return jaccard_target_df, cosine_target_df

    def compute_movie_user_similarities(self):
        start = time()
        movie_indices = self.movie_terms_df.index.values
        user_indices = self.user_terms_df.index.values

        movie_indices.sort()
        user_indices.sort()

        cosine_similarities = {}
        jaccard_similarities = {}

        for movie_id in movie_indices:
            #     other_movies = np.setdiff1d(movie_indices, movie1_id)

            for user_id in user_indices:
                movie_genre = self.movie_terms_df.loc[movie_id, '1':'19'].astype(np.int64)
                user_terms = self.user_terms_df.loc[user_id, '1':'19'].astype(np.int64)

                jaccard_similarity = jaccard(movie_genre, user_terms)
                cosine_similarity = cosine(movie_genre, user_terms)

                if movie_id not in cosine_similarities:
                    cosine_similarities[movie_id] = {}
                if user_id not in cosine_similarities:
                    cosine_similarities[user_id] = {}

                if movie_id not in jaccard_similarities:
                    jaccard_similarities[movie_id] = {}
                if user_id not in jaccard_similarities:
                    jaccard_similarities[user_id] = {}

                cosine_similarities[movie_id][user_id] = cosine_similarity
                cosine_similarities[user_id][movie_id] = cosine_similarity

                jaccard_similarities[movie_id][user_id] = jaccard_similarity
                jaccard_similarities[user_id][movie_id] = jaccard_similarity

        movie_user_jaccard_similarities_df = pd.DataFrame(jaccard_similarities)
        movie_user_cosine_similarities_df = pd.DataFrame(cosine_similarities)

        finish = time() - start
        self.verbose and print("Movie-User similarities calculation time: %f seconds" % finish)

        return movie_user_jaccard_similarities_df, movie_user_cosine_similarities_df

    def get_k_nearest_users(self, similarity_function, k, active_user_id, candidate_movie_id=None):
        if self.user_user_all_similarities_df is None:
            if not self.read_from_pickle:
                self.user_user_all_similarities_df = self.fast_compute_entity_entity_similarities(self.user_terms_df,
                                                                                                  self.user_user_all_similarities_df)
            else:
                self.compute_similarities()

        tuple_index = self.get_tuple_index_for_similarity_measure(similarity_function)

        nearest_users = self.user_user_all_similarities_df.loc[:, active_user_id].dropna().apply(
            lambda x: x[tuple_index])
        nearest_users.sort_values(ascending=False, inplace=True)

        return nearest_users.index.tolist()[:k]

    def get_k_nearest_movies(self, similarity_function, k, candidate_movie_id, active_user_id=None):
        if self.movie_movie_all_similarities_df is None:
            if self.read_from_pickle:
                self.movie_movie_all_similarities_df = self.fast_compute_entity_entity_similarities(self.movie_terms_df,
                                                                                                    self.movie_movie_all_similarities_df)
            else:
                self.compute_similarities()

        tuple_index = self.get_tuple_index_for_similarity_measure(similarity_function)

        nearest_movies = self.movie_movie_all_similarities_df.loc[:, candidate_movie_id].dropna().apply(
            lambda x: x[tuple_index])
        nearest_movies.sort_values(ascending=False, inplace=True)

        nearest_movie_ids = nearest_movies.index.tolist()[:k]
        nearest_movie_similarities = nearest_movies.tolist()[:k]
        nearest_movie_ratings = self.movie_ratings_all_users_avg[nearest_movie_ids].tolist()[:k]

        return nearest_movie_ids, nearest_movie_similarities, nearest_movie_ratings

    def get_k_most_similar_movies(self, k, user_id, similarity_measure):
        if self.movie_user_all_similarities_df is None:
            self.fast_compute_movie_user_similarities()

        tuple_index = self.get_tuple_index_for_similarity_measure(similarity_measure)

        nearest = self.movie_user_all_similarities_df.loc[:, user_id].apply(lambda x: x[tuple_index])
        nearest.sort_values(ascending=False, inplace=True)

        movies_scores_dict = {}
        for a in nearest.groupby(nearest):
            movies_scores_dict[a[0]] = pd.Series(a[1].index.values)

        nearest_df = pd.DataFrame(movies_scores_dict)
        nearest_df_index = nearest_df.columns.values

        nearest_df_index[::-1].sort()
        all_nearest_movies = list()

        # Ties in similarity should be broken based on average
        #   rating of the item across all users.
        for index, corr_score in enumerate(nearest_df_index):
            # for each unique value in score, get list of all nearest movies to the user with this score
            correlation_ranks_array = nearest_df.loc[:, nearest_df_index[index]].dropna().values.astype(
                np.int64).tolist()

            # key=self.sorted_all_user_avg_ratings.index
            # key attribute makes sure that the order retained will be as in the list 'self.sorted_all_user_avg_ratings'
            # i.e. movie with highest all user's average rating on top
            intersection = sorted(set(self.sorted_all_user_avg_ratings) & set(correlation_ranks_array),
                                  key=self.sorted_all_user_avg_ratings.index)

            # the remaining movies with this score
            difference = sorted(set(correlation_ranks_array) - set(intersection), key=correlation_ranks_array.index)

            all_nearest_movies.extend(intersection)
            all_nearest_movies.extend(difference)

        k_most_similar_movies = all_nearest_movies[:k]
        # nearest_movie_similarities = self.movie_user_all_similarities_df.loc[k_most_similar_movies, movie_id].apply(
        #     lambda x: x[tuple_index]).values

        return k_most_similar_movies, self.movie_terms_df.loc[
            k_most_similar_movies, 'name'].tolist()

    def get_k_most_similar_movies_rated_by_user(self, k, user_id, movie_id, similarity_measure):
        if self.movie_user_all_similarities_df is None:
            self.fast_compute_movie_user_similarities()
        if self.movie_movie_all_similarities_df is None:
            if not self.read_from_pickle:
                self.movie_movie_all_similarities_df = self.fast_compute_entity_entity_similarities(self.movie_terms_df,
                                                                                                    self.movie_movie_all_similarities_df)
            else:
                self.compute_similarities()

        tuple_index = self.get_tuple_index_for_similarity_measure(similarity_measure)

        # nearest = self.movie_user_all_similarities_df.loc[:, user_id].apply(lambda x: x[tuple_index])
        nearest = self.movie_movie_all_similarities_df.loc[:, movie_id].dropna().apply(lambda x: x[tuple_index])
        movies_rated_by_user = self.movie_ratings_df.loc[:, user_id].dropna()
        # nearest = self.movie_movie_all_similarities_df.reindex(movie_id, axis=1).dropna().apply(lambda x: x[tuple_index])
        # movies_rated_by_user = self.movie_ratings_df.reindex(user_id, axis=1).dropna()
        nearest = nearest[movies_rated_by_user.index].dropna()
        nearest.sort_values(ascending=False, inplace=True)

        movies_scores_dict = {}
        for a in nearest.groupby(nearest):
            movies_scores_dict[a[0]] = pd.Series(a[1].index.values)

        nearest_df = pd.DataFrame(movies_scores_dict)
        # TODO know why having this print statements gives expected results otherwise they're messed up
        # TODO remove later
        #  ------------
        current_stdout = sys.stdout

        class ListStream:
            def __init__(self):
                self.data = []

            def write(self, s):
                self.data.append(s)

        sys.stdout = x = ListStream()
        print(nearest_df)
        sys.stdout = current_stdout
        # TODO remove later
        #  ------------

        nearest_df_index = nearest_df.columns.values
        nearest_df_index[::-1].sort()
        all_nearest_movies = list()

        # Ties in similarity should be broken based on average
        #   rating of the item across all users.
        for index, corr_score in enumerate(nearest_df_index):
            # for each unique value in score, get list of all nearest movies to the user with this score
            correlation_ranks_array = nearest_df.loc[:, nearest_df_index[index]].dropna().values.astype(
                np.int64).tolist()

            # key=self.sorted_all_user_avg_ratings.index
            # key attribute makes sure that the order retained will be as in the list 'self.sorted_all_user_avg_ratings'
            # i.e. movie with highest all user's average rating on top
            intersection = sorted(set(self.sorted_all_user_avg_ratings) & set(correlation_ranks_array),
                                  key=self.sorted_all_user_avg_ratings.index)

            # the remaining movies with this score
            difference = sorted(set(correlation_ranks_array) - set(intersection), key=correlation_ranks_array.index)

            all_nearest_movies.extend(intersection)
            all_nearest_movies.extend(difference)

        nearest_movie_ids = all_nearest_movies[:k]
        nearest_movie_similarities = self.movie_movie_all_similarities_df.loc[nearest_movie_ids, movie_id].apply(
            lambda x: x[tuple_index]).values
        nearest_movie_ratings = self.movie_ratings_all_users_avg[nearest_movie_ids].values[:k]

        return nearest_movie_ids, nearest_movie_similarities, nearest_movie_ratings

    def get_tuple_index_for_similarity_measure(self, similarity_measure):
        # Jaccard similarity is at index 0 in tuple
        # Cosine similarity is at index 1 in tuple
        tuple_index = 0

        if similarity_measure == 'cosine':
            tuple_index = 1

        return tuple_index


class PredictionsForContentBased:

    def __init__(self, movies_file, user_terms_file, user_ratings_file, verbose_flag=False, save_pickle=False,
                 read_from_pickle=False,
                 pickle_dir=""):
        self.movies_file = movies_file
        self.user_terms_file = user_terms_file
        self.user_ratings_file = user_ratings_file

        # TODO remove unused fields
        self.user_terms_df = None
        self.movie_terms_df = None
        self.movie_ratings_df = None
        self.movie_ratings_all_users_avg = None
        self.sorted_all_user_avg_ratings = None
        self.movie_movie_all_similarities_df = None
        self.user_user_all_similarities_df = None
        self.movie_user_all_similarities_df = None
        self.movie_movie_jaccard_similarities_df = None
        self.movie_movie_cosine_similarities_df = None
        self.movie_user_jaccard_similarities_df = None
        self.movie_user_cosine_similarities_df = None
        self.verbose = verbose_flag
        self.content_based_recommender = None

        self.known_movie_ratings = None
        self.movie_ratings_train = None
        self.movie_ratings_test = None

        self.known_movie_class_train = None
        self.known_movie_class_test = None
        self.known_movie_class = None

        self.test_ratings = None

        self.save_pickle = save_pickle
        self.read_from_pickle = read_from_pickle
        self.pickle_dir = pickle_dir

    def make_predictions(self, k=50, similarity_measure='jaccard', test_size_frac=0.005):
        self.content_based_recommender = Content_Based_Recommender(movies_file=self.movies_file,
                                                                   user_terms_file=self.user_terms_file,
                                                                   user_ratings_file=self.user_ratings_file,
                                                                   verbose_flag=self.verbose,
                                                                   save_pickle=self.save_pickle,
                                                                   read_from_pickle_flag=self.read_from_pickle,
                                                                   pickle_dir=self.pickle_dir)

        # movies_file, user_terms_file, user_ratings_file, verbose_flag = False, save_pickle = False,
        # read_from_pickle_flag = False,
        # pickle_dir = ""
        self.content_based_recommender.load_ratings()

        df = pd.DataFrame(None)
        result = self.get_accuracy_report(k, similarity_measure, test_size_frac)
        df = self.update_report(k, similarity_measure, test_size_frac, result, df)
        df['test_data_size'] = df['test_size_frac'].values * 100000

        df.apply(lambda x: print(x), axis=1)

    def update_report(self, k, similarity_measure, test_size_frac, result, df):
        series = pd.Series(None)
        series['k'] = k
        series['similarity_measure'] = similarity_measure
        series['test_size_frac'] = test_size_frac
        series['bias'] = result[1]
        series['mae'] = result[0][0]
        series['rmse'] = result[0][1]
        series['precision'] = result[0][2]
        series['recall'] = result[0][3]
        series['conf_matrix'] = result[0][4]

        df = df.append(series, ignore_index=True)

        return df

    def get_accuracy_report(self, k, similarity_measure, test_size_frac):
        rows_n_columns = pd.read_csv(filepath_or_buffer=self.user_ratings_file, sep="\s+", usecols=range(3),
                                     names=["userid", "movieid", "rating"])
        test_ratings = rows_n_columns.sample(frac=test_size_frac)
        test_ratings.set_index(test_ratings['userid'], drop=True, inplace=True)
        # print(test_ratings)
        test_ratings['regression'] = np.nan

        # for classification accuracy
        # known_movie_class = pd.qcut(test_ratings['rating'], q=2, labels=['Like', 'Dislike'])
        known_movie_class = test_ratings.apply(lambda x: 'Like' if x['rating'] >= 2.5 else 'Dislike', axis=1)
        # known_movie_class_train, known_movie_class_test = train_test_split(known_movie_class, test_size=0.3)
        test_ratings['actual_class'] = known_movie_class

        def update_rating(self, series):
            predicted_rating = 0

            # predict rating here
            userid = int(series['userid'])
            movieid = int(series['movieid'])

            # get user's k most similar movies
            # get ratings for those movies
            # get movie-movie with movieid similarities for those movies
            #     nearest_movie_ids, nearest_movie_similarities, nearest_movie_ratings = get_k_nearest_movies(similarity_function, k, movieid)
            nearest_movie_ids, nearest_movie_similarities, nearest_movie_ratings = \
                self.content_based_recommender.get_k_most_similar_movies_rated_by_user(k, userid, movieid,
                                                                                       similarity_measure)

            nearest_movie_ids, nearest_movie_similarities, nearest_movie_ratings = \
                np.array(nearest_movie_ids), np.array(nearest_movie_similarities), np.array(nearest_movie_ratings)

            weights = np.divide(1, np.square(nearest_movie_similarities))

            # aggregate these ratings with weights
            #     predicted_rating = np.sum(np.multiply(nearest_movie_ratings, nearest_movie_similarities)) / np.sum(nearest_movie_similarities)
            predicted_rating = np.sum(np.multiply(nearest_movie_ratings, weights)) / np.sum(weights)

            predicted_class = 'Dislike'
            if predicted_rating >= 2.5:
                predicted_class = 'Like'

            # predict the rating for series
            series['regression'] = predicted_rating
            series['classification'] = predicted_class

            return series

        start = time()
        test_ratings = test_ratings.apply(lambda x: update_rating(self, x), axis=1)
        # test_ratings.sort_values(test_ratings.index)
        test_ratings.reset_index(drop=True, inplace=True)
        finish = time() - start

        print("Time taken %f seconds" % finish)
        return self.get_accuracy_scores(test_ratings), self.get_system_bias(test_ratings)

    def get_accuracy_scores(self, test_ratings):
        trimmed_ratings = test_ratings.dropna()

        regression_rating = trimmed_ratings['regression'].values
        actual_rating = trimmed_ratings['rating'].values

        #     classification = test_ratings['classification'].values
        #     actual_class = test_ratings['actual_class'].values

        classification = trimmed_ratings['classification'].values
        actual_class = trimmed_ratings['actual_class'].values

        def rmse(predictions, targets):
            return np.sqrt(((predictions - targets) ** 2).mean())

        mae = mean_absolute_error(actual_rating, regression_rating)
        rmse = rmse(actual_rating, regression_rating)
        #     classification_rep = classification_report(actual_class, classification, labels=['Like', 'Dislike'], output_dict=True)
        conf_matrix = confusion_matrix(actual_class, classification, labels=['Like', 'Dislike'])
        precision = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[1][0])
        recall = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])

        return mae, rmse, precision, recall, conf_matrix

    def get_system_bias(self, test_ratings):
        trimmed_ratings = test_ratings.dropna()

        result = pd.DataFrame(
            trimmed_ratings.apply(lambda x: "over" if x['regression'] > x['rating'] else "under", axis=1))
        under_predictions = len(result[result[0] == 'under'])
        over_predictions = len(result[result[0] == 'over'])

        if under_predictions > over_predictions:
            bias = 'Under Predictions'
        elif under_predictions < over_predictions:
            bias = 'Over Predictions'
        else:
            bias = 'Equally Balanced'

        return bias


class Hybrid_Recommender:

    def __init__(self, user_system_preference, collaborative_filtering_system, content_based_system,
                 switching_probability=0.4, alpha_value=0.4, beta_value=0.6, k=500, movie_id=1, user_id=1):
        self.user_preference = user_system_preference
        self.switching_probability = switching_probability
        self.alpha_value = alpha_value
        self.beta_value = beta_value
        self.collaborative_filtering_system = collaborative_filtering_system
        self.content_based_system = content_based_system
        self.k = k
        self.movie_id = movie_id
        self.user_id = user_id

    def run_hybrid_recommender(self):
        if self.user_preference is "weighting":
            self.weighting_for_recommenders()
        else:
            self.switch_recommender()

    def weighting_for_recommenders(self):
        k = 500

        # user_id = 103
        # movie_id = 202
        user_id = 1
        movie_id = 15

        # TODO choose such user id and movie id that yield some movies common in intersection
        # self.collaborative_filtering_system.
        # get k nearest movies from collaborative filtering using pearson similarity measure
        result_collaborative = self.collaborative_filtering_system.get_k_nearest_movies(
            pearson_correlation, k, movie_id, user_id)
        # k_movies_from_content_based = self.content_based_system.get_k_most_similar_movies(k=k, user_id=user_id,
        #                                                                                   similarity_measure="jaccard")

        # get k most similar movies from content based recommender using jaccard similarity measure
        result_content_based = self.content_based_system.get_k_most_similar_movies_rated_by_user(k, user_id,
                                                                                                 movie_id,
                                                                                                 similarity_measure="jaccard")

        movies1 = result_collaborative[0].tolist()
        similarities1 = result_collaborative[1].tolist()

        movies2 = result_content_based[0]
        similarities2 = result_content_based[1]

        # get common movies and their scores through manual intersection
        intersection = sorted(set(movies1) & set(movies2), key=movies1.index)

        list1_index = list()
        list2_index = list()
        for i in intersection:
            list1_index.append(movies1.index(i))
            list2_index.append(movies2.index(i))

        similarities1_list = list()
        for i in list1_index:
            similarities1_list.append(similarities1[i])

        similarities2_list = list()
        for i in list2_index:
            similarities2_list.append(similarities2[i])

        the_dictionary = {'Product': intersection, 'Score1': similarities1_list, 'Score2': similarities2_list}
        # properly organize all data in DataFramme
        df = pd.DataFrame(the_dictionary)

        # compute ranks for score1 and score2
        df['Rank1'] = df['Score1'].rank(method='first', ascending=False).astype(np.int64)
        df['Rank2'] = df['Score2'].rank(method='first', ascending=False).astype(np.int64)

        # compute compound score using alpha and beta values of self obj
        df['Compound_Score'] = self.alpha_value * df['Score1'].values + self.beta_value * df['Score2'].values

        # compute ranks for compound score i.e. Hybrid Rank
        df['Hybrid_Rank'] = df['Compound_Score'].rank(method='first', ascending=False).astype(np.int64)

        # calculate ranking accuracy (part 3)
        self.ranking_accuracy(df)

    def ranking_accuracy(self, df):
        # spearman ranking
        rank1 = df['Rank1'].values
        rank2 = df['Rank2'].values
        n = len(rank1)

        ranking_accuracy = 1 - (6 * np.sum(np.square(rank1 - rank2))) / (n * (np.square(n) - 1))
        print("Ranking Accuracy (Spearman Ranking): %f" % (ranking_accuracy * 100) + "%")
        return ranking_accuracy * 100

    def switch_recommender(self):
        switching_probability = random()

        if switching_probability <= self.switching_probability:
            self.choose_collaborative_filtering_recommender()
        else:
            self.choose_content_based_recommender()

    def choose_content_based_recommender(self):
        self.content_based_run_for_hybrid()

    def choose_collaborative_filtering_recommender(self):
        self.collaborative_based_run_for_hybrid()

    def content_based_run_for_hybrid(self):
        # here goes the code for running some experiments on content based system
        print("Running Hybrid - Content Based Recommender System with k=20")
        k = 20
        result = self.content_based_system.get_k_most_similar_movies(k, self.user_id,
                                                            # self.movie_id,
                                                            similarity_measure="jaccard")
        print("Movie ID's:\n" + str(result[0]))
        print("\nMovie names:\n" + str(result[1]))

    def collaborative_based_run_for_hybrid(self):
        # here goes the code for running some experiments on collaborative filtering based system
        print("Running Hybrid - Collaborative Filtering Recommender System with k=20")
        k = 20
        result = self.collaborative_filtering_system.get_k_nearest_movies(pearson_correlation, k, self.movie_id, self.user_id)
        print("Movie ID's:\n" + str(result[0]))
        print("\nSimilarities:\n" + str(result[1]))


def cosine_similarity(x, y):
    return 1 - cosine(x, y)


def adjusted_cosine_similarity(x, y):
    """duplicated just to avoid few extra checks and additional code to get mean centered values"""
    return 1 - cosine(x, y)


def pearson_correlation(x, y):
    return pearsonr(x, y)[0]


def main_collaborative(rc):
    start_time = time()

    # # ratings_file = "./tempInput/exercise_example"
    # ratings_file = "./input/ratings_1.txt"
    #
    # save_pickle = False
    # read_from_pickle = True
    # pickle_dir = './input/pickle/collaborative/'
    #
    #
    # rc = Collaborative_Filtering_Recommender(read_from_pickle_flag=read_from_pickle, save_pickle_flag=save_pickle,
    #                                          pickle_dir=pickle_dir)

    k = 20
    active_user_id = 103
    candidate_movie_id = 202
    threshold = 0

    print(rc.get_k_nearest_users(pearson_correlation, k, active_user_id, candidate_movie_id))
    print(rc.get_k_thresholded_nearest_users(pearson_correlation, k, threshold, active_user_id, candidate_movie_id))
    print(rc.get_k_nearest_movies(pearson_correlation, k, candidate_movie_id, active_user_id))
    print(rc.get_k_thresholded_nearest_movies(pearson_correlation, k, threshold, candidate_movie_id, active_user_id))

    # print(rc.get_k_thresholded_nearest_movies(adjusted_cosine_similarity, k, threshold, candidate_movie_id, None))

    finish_time = time() - start_time
    print("\n\nTotal Time Taken: %f seconds" % finish_time)


def main_content_based(content_based_recommender):
    # TODO run multiple experiments as mentioned in part 1 description
    k = 5
    user_id = 450
    similarity_measure = 'jaccard'
    # similarity_measure = 'cosine'

    start = time()

    content_based_recommender.load_ratings()
    content_based_recommender.compute_similarities()
    k_similar_movie_ids, k_similar_movie_names = content_based_recommender.get_k_most_similar_movies(k=k,
                                                                                                     user_id=user_id,
                                                                                                     similarity_measure=similarity_measure)
    print("K similar movies: " + str(k_similar_movie_names))
    print("\nUser 10 Cosine:\n" + str(
        content_based_recommender.get_k_most_similar_movies(k, user_id=10, similarity_measure="cosine")))
    print(
        "\nUser 10 Jaccard:\n" + str(
            content_based_recommender.get_k_most_similar_movies(k, user_id=10, similarity_measure="jaccard")))
    results = content_based_recommender.get_k_most_similar_movies_rated_by_user(k, user_id=450, movie_id=1,
                                                                                similarity_measure=similarity_measure)
    print(results)

    finish = time() - start

    print("\n\nTotal Time: %f seconds" % finish)


def main_predictions(predictions):
    predictions.make_predictions()


def main_hybrid(collaborative_filtering_recommender, content_based_recommender, k=500):
    # user_system_preference = 'weighting' or 'switching'
    # collaborative_filtering_system, content_based_system, switching_probability = 0.4, alpha_value = 0.4, beta_value = 0.6
    print("\nHybrid - Weighting:\n")
    hybrid_recommender_system = Hybrid_Recommender("weighting", collaborative_filtering_recommender,
                                                   content_based_recommender, switching_probability=0.4,
                                                   alpha_value=0.4, beta_value=0.6, k=k)
    hybrid_recommender_system.run_hybrid_recommender()

    print("\nHybrid - Switching:\n")
    hybrid_recommender_system = Hybrid_Recommender("switching", collaborative_filtering_recommender,
                                                   content_based_recommender, switching_probability=0.4,
                                                   alpha_value=0.4, beta_value=0.6, k=k)
    hybrid_recommender_system.run_hybrid_recommender()


if __name__ == '__main__':
    # create object of all systems here and pass them to hybrid
    user_terms_file = "./input/userterms.txt"
    movies_file = "./input/movies.txt"
    ratings_file = './input/ratings_1.txt'
    #     user_terms_file = "./micro_input/userterms.txt"
    #     movies_file = "./micro_input/movies.txt"
    #     ratings_file = './micro_input/ratings_1.txt'

    verbose_flag = False

    # instead of computing all similarities, they will be loaded from the pickle file
    # TODO decide whether to keep this flag on or off  while submitting the assignment - make it false maybe
    # Only either of these should be True at at a time, or both False
    save_pickle = False
    read_from_pickle = True

    pickle_dir = './input/pickle/'

    start = time()
    # Content Based Recommender System - Experimentation Results
    content_based_recommender = Content_Based_Recommender(movies_file, user_terms_file, ratings_file, verbose_flag,
                                                          save_pickle,
                                                          read_from_pickle, pickle_dir)
    content_based_recommender.load_ratings()

    pickle_dir_collaborative = './input/pickle/collaborative/'

    threshold = 0

    # Collaborative Filtering output
    collaborative_filtering = Collaborative_Filtering_Recommender(user_ratings_file=ratings_file,
                                                                  read_from_pickle_flag=read_from_pickle,
                                                                  save_pickle_flag=save_pickle,
                                                                  pickle_dir=pickle_dir_collaborative)
    collaborative_filtering.load_ratings()

    # Predictions, Classification and Bias Detection for Content Based Recommender System
    predictions = PredictionsForContentBased(movies_file=movies_file, user_terms_file=user_terms_file,
                                             user_ratings_file=ratings_file, save_pickle=save_pickle,
                                             read_from_pickle=read_from_pickle, pickle_dir=pickle_dir,
                                             verbose_flag=verbose_flag)

    # Part a
    # Content Based Recommender System - Experimentation Results
    print("#------------------- Content Based Recommender System - Experimentation Results -------------------#\n")
    main_content_based(content_based_recommender)
    #
    # # Part b and ranking accuracy from Part c
    print("\n\n#------------------- Hybrid Recommender System - Experimentation Results -------------------#")
    main_hybrid(collaborative_filtering, content_based_recommender, k=50)

    #
    # # Part c
    print("\n\n#------ Predictions, Classification and Bias Detection for Content Based Recommender System ------#\n")
    main_predictions(predictions)
