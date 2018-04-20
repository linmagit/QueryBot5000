#!/usr/bin/env python
# coding: utf-8

import re
import string
import numpy as np


# Create the logical vectors for templates
def create_vectors(templates, semantics_dict):

    ##################################################################
    # Variable initialization, and defining constants for use in our #
    # feature vector creation                                        #
    ##################################################################
    data = []
    # Special SQL-keywords
    query_type = ["SELECT", "INSERT", "UPDATE", "DELETE"]
    # INSERT, UPDATE and DELETE are closer to one another than to SELECT.
    keywords = ["GROUP", "ORDER", "HAVING", "LIMIT"]
    # keywords are wrapped up into SELECT

    # Don't change the next few lines
    tables = list(semantics_dict)
    columns = extract_columns(semantics_dict)
    num_tables = len(tables)
    num_columns = len(columns)

    # Result vector dictionary
    vector_dict = dict()

    ###################################################################
    # Vector Format                                                   #
    # [query type, table_dist, table_1, table_2, table_3..., table_n, #
    # column_1, column_2, ... , column_m]                             #
    ###################################################################
    for template in templates:

        vector_length = 2 + num_tables + num_columns
        vector = np.zeros(vector_length)

        #print(template)

        query = re.split('\s+', template) # split query by whitespace
        for sem in query:
            sem = str(sem)
            #print(sem)

            if sem in string.whitespace:  # Verified correct
                continue

            if sem.upper() in query_type:  # Verified correct
                sem = sem.upper()
                # This enforces the distance between query_type
                index = query_type.index(sem)
                #print("type?", index)
                if index != 0:
                    vector[0] = index + 5
                    continue
                else:
                    vector[0] = index  # 0
                    continue

            elif sem.upper() in keywords:
                sem = sem.upper()
                # since we know that the keywords are a byproduct
                # of SELECT, since SELECT is 0, we modify on the first
                # index
                vector[0] += keywords.index(sem)
                continue

            elif sem in columns:
                #print("col1: ", sem)
                vector[2 + num_tables + columns.index(sem)] += 1
                continue

            else:  # Handles cases of tables + useless stuff
                try:
                    # Since iterating through log, will come across
                    # words that are irrelevant. This allows us to remove
                    # them in constant time
                    semantics_dict[sem]
                    table_index = tables.index(sem)
                    #print(template)
                    #print("table: ", sem)
                    vector[2 + table_index] += 1
                    continue
                except:
                    # Meaning that it's not a table, but it could be a table.col
                    # format. Yeah, fuck you too sql
                    try:
                        (table, col) = re.split("[.,]", sem)[:2]
                    except:
                        continue
                    if table in tables:
                        table_index = tables.index(table)
                    else:
                        table_index = -1
                    if col in columns:
                        col_index = columns.index(col)
                        #print("col2: ", col)
                    else:
                        col_index = -1

                    if table_index != -1:
                        vector[2 + table_index] += 1
                    if col_index != -1:
                        vector[2 + num_tables + col_index] += 1
                    continue

        vector_dict[template] = vector
        #print(vector)

    print("Built Feature Vectors...")
    return vector_dict


def delete_zero_cols(arr):
    """ Deletes columns that are entirely zero (dimension reduction)
    Args:
        arr (numpy array): feature vectors
    Returns:
        (numpy array): reduced feature vectors
    """

    zero_cols = np.nonzero(arr.sum(axis=0) == 0)
    arr = np.delete(arr, zero_cols, axis=1)
    return arr


def extract_columns(semantics_dict):
    """Extracts the columns for each table (for the inner dictionary)
    from the semantic dictionary

    Args:
       semantics_dict (dict): dict of dicts

    Returns:
        (list): union of all columns from all tables

    """
    columns = set()
    for table in semantics_dict:
        column_dict = semantics_dict[table]
        columns = columns.union(set(column_dict))
    return list(columns)
