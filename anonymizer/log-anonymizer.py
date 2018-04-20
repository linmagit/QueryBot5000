#!/usr/bin/env python3.5

import sys
import re
import gzip
import csv
import sqlparse
import hashlib
import string
import logging
import argparse
import zipfile

from pprint import pprint
global ANONYMIZE

# ==============================================
# LOGGING CONFIGURATION
# ==============================================

LOG = logging.getLogger(__name__)
LOG_handler = logging.StreamHandler()
LOG_formatter = logging.Formatter(
    fmt='%(asctime)s[%(funcName)s:%(lineno)03d]%(levelname)-5s:%(message)s',
    datefmt='%m-%d-%Y %H:%M:%S')
LOG_handler.setFormatter(LOG_formatter)
LOG.addHandler(LOG_handler)
LOG.setLevel(logging.INFO)

CMD_TYPES = [
    "Connect",
    "Quit",
    "Init DB",
    "Query",
    "Field List",
    "Statistics"
]
# SQL commands that we just want to simply ignore
IGNORED_CMDS = []

CLEAN_CMDS = [
    re.compile(r"(WHERE|ON)[\s]{2,}", re.IGNORECASE)
]

# <DATE>T<TIME>Z<CONNECTION> <CMD> <QUERY>
# 2016-09-01T15:24:06.654226Z111282 Query SHOW CREATE TABLE `XXX`.`YYY`
# 2016-11-15T20:06:50.096423Z        11 Query     SELECT DATABASE()
LOG_REGEX = re.compile(
    "([\d]{4}-[\d]{2}-[\d]{2})T([\d]{2}:[\d]{2}:[\d]{2}\.[\d]+)Z[\s]*([\d]+)[\s]+(" +
    "|".join(CMD_TYPES) + ")[\s]+(.*)")


LOG_REGEX_POSTGRESQL = re.compile(
    "([\d]{4}-[\d]{2}-[\d]{2} [\d]{2}:[\d]{2}:[\d]{2}.[\d]*" + 
    " (EST|EDT)),\".*?\",\".*?\",([\d]+),\".*?\",.*?,\".*?\",([\d]{4}-[\d]{2}-[\d]{2}"
    " [\d]{2}:[\d]{2}:[\d]{2} (EST|EDT)),.*?,\d,LOG,[\d]{5},\"(.*?)\",+?\"(.*?)\".*")

POSTGRESQL_PREFIX_REGEX = re.compile("[\d]{4}-[\d]{2}-[\d]{2} [\d]{2}:[\d]{2}:[\d]{2}.[\d]*")

# Regex for MySQL 5.5
# 170214  9:01:51 1496503 Query   SELECT t0__Section.guid FROM section t0__Section WHERE ***
#                 1496503 Query   SELECT admit_code, title, start_date, end_date, time_zone, ***
LOG_REGEX_5_5 = re.compile(
        "(\d{6}\s+\d+:\d+:\d+)?\s+([\d]+)[\s]+(" +
    "|".join(CMD_TYPES) + ")[\s]+(.*)")


# SELECT <TARGET> FROM <TABLE> ....""
QUERY_REGEX = re.compile(r"(\bSELECT\b|\bselect\b)[\s]+(\S*)[\s]+" +
                         r"(\bFROM\b|\bfrom\b)[\s]+(\S*)[\s]+.*")

OUTPUT = csv.writer(sys.stdout, quoting=csv.QUOTE_ALL)


# ==============================================
# anonymize
# ==============================================


def is_date_or_digits(inspect):
    """
    :p inspect: the element that we need to inspect
    :t inspect: string
    returns true if it fits our idea of an integer, or a date
    dates fit the format of <YEAR>-<MONTH>-<DAY> (month & day are 2 chars)
    """
    inspect = inspect[1: -1]
    try:
        inspect = int(inspect)
        return True
    except:
        if len(inspect) != 10 and len(inspect) != 19:
            return False
        elif inspect[4] != "-" and inspect[7] != "-":
            return False
        else:
            try:
                int(inspect[0:4])
                int(inspect[5:7])
                int(inspect[8:10])
                return True
            except:
                return False

##########################
# # Different extractors #
##########################

def iterator_extractor(to_extract, holder, salt):
    """
    :p to_extract: our tokens containing our content to
                   potentially salt + hash
    :t to_extract: iterator
    :p holder: the accumulator of our processed query
    :t holder: string
    :p salt: the salt key
    :t salt: string
    returns the modified holder
    """
    for val in to_extract:
        val = str(val)
        if ((val in string.whitespace) or (val in string.punctuation) or
                is_date_or_digits(val)):
            holder = holder + " " + val
        else:
            data_length = str(len(val))  # to account for " & "
            cleaned = hashlib.md5(
                salt + val.encode("utf-8")).hexdigest()
            holder = holder + " '" + (data_length + "\\" + str(cleaned)) + "'"
    return holder


def token_extractor(to_extract, holder, salt):
    """
    :p to_extract: our tokens containing our content to
                   potentially salt + hash
    :t to_extract: Token (from sqlparse)
    :p holder: the accumulator of our processed query
    :t holder: string
    :p salt: the salt key
    :t salt: string
    returns the modified holder
    """
    to_extract = str(to_extract)
    if (to_extract in string.whitespace or to_extract in string.punctuation
            or is_date_or_digits(to_extract)):
        holder = holder + to_extract
    else:
        data_length = str(len(to_extract) - 2)  # to account for " & "
        cleaned = hashlib.md5(
            salt + to_extract.encode("utf-8")).hexdigest()
        holder = holder + " '" + (data_length + "\\" + str(cleaned)) + "'"
    return holder


def parenthesis_extractor(to_extract, holder, salt):
    """
    :p to_extract: our tokens containing our content to
                   potentially salt + hash
    :t to_extract: Token (from sqlparse)
    :p holder: the accumulator of our processed query
    :t holder: string
    :p salt: the salt key
    :t salt: string
    returns the modified holder
    """
    for val in to_extract:
        if isinstance(val, sqlparse.sql.IdentifierList):
            holder = iterator_extractor(val, holder, salt)
        elif isinstance(val, sqlparse.sql.Token):
            holder = token_extractor(val, holder, salt)
        else:
            holder = holder + str(val)
    return holder


def comparison_extractor(to_extract, holder, salt):
    """
    :p to_extract: our tokens containing our content to
                   potentially salt + hash
    :t to_extract: Token (from sqlparse)
    :p holder: the accumulator of our processed query
    :t holder: string
    :p salt: the salt key
    :t salt: string
    returns the modified holder
    """
    elem = str(to_extract)
    for delim in ("=", ">", ">=", "<", "<=", "!=", "LIKE"):
        splitter = elem.find(delim)
        if splitter == -1:
            continue
        notPrivate = elem[0:splitter + 1]
        private = elem[splitter + 1:]
        if is_date_or_digits(private):
            holder = holder + " " + notPrivate + " " + private
        else:
            private = private.strip()  # heh heh
            data_length = str(len(private) - 2)
            cleaned = hashlib.md5(salt +
                                  private.encode("utf-8")).hexdigest()
            holder = holder + " " + notPrivate + " '" + \
                (data_length + "\\" + str(cleaned)) + "'"
    return holder
    # FOR
    raise Exception("Error in indexing of where statement: %s" % elem)
# DEF

##########################
# # Processing functions #
##########################


def hash_and_salt(to_clean, flag):
    """
    :p to_clean: sensitive data
    :t to_clean: string
    returns a salted query
    """
    holder = ""
    if flag == 1:  # WHERE
        # PAVLO
        # I fixed the casting of 'to_clean' to string so that we can
        # extract substrings properly.
        holder = holder + str(to_clean)[0:5] + " "
        to_clean = to_clean[1:]
    elif flag == 2:  # COMPARISON
        # we hit a comparison before going into the individual tokens
        return comparison_extractor(to_clean, holder, SALT)
    if str(to_clean).upper().find("LIKE") != -1:
        for elem in to_clean:
            if isinstance(elem, sqlparse.sql.Identifier):
                holder = holder + str(elem)
            else:
                elem = str(elem)
                if elem in string.whitespace:
                    holder = holder + elem
                elif elem.find('%') != -1 or elem.find('_') != -1:
                    holder = (holder + "'" + str(len(elem)) + "\\" +
                              hashlib.md5(SALT + elem.encode("utf-8")).hexdigest() + "'")
                else:
                    holder += elem
        # FOR
        return holder
    # IF
    for elem in to_clean:
        if isinstance(elem, sqlparse.sql.Comparison):
            holder = comparison_extractor(elem, holder, SALT)
        # SQL COMPARISON
        elif isinstance(elem, sqlparse.sql.TokenList):
            if (isinstance(elem, sqlparse.sql.Function)):
                holder = holder + str(elem)
            elif isinstance(elem, sqlparse.sql.Parenthesis):
                holder = parenthesis_extractor(elem, holder, SALT)
            else:
                holder = iterator_extractor(elem, holder, SALT)
        # SQL TOKENLIST
        elif isinstance(elem, sqlparse.sql.Identifier):
            holder = iterator_extractor(elem, holder, SALT)
        # SQL IDENTIFIER
        else:
            holder = holder + str(elem)
    return holder


def anonymize(sql):
    """
    "SELECT content from content where name = 'Instructions' AND domain_id = 1"
    :p sql : a SQL query from a sql general log file
    :t sql : string
    returns a query with the non-dates and non-numerics-only
    salted, and pre-pended with the length
    """
    to_clean = None
    # If we can't parse it, then it is not of the form we want
    # e.g it is just a Quit statement or something
    try:
        to_clean = sqlparse.parse(sql)[0].tokens
    except:
        return sql
    # Actual parsing
    clean_sql = ""
    for unclean in to_clean:
        if isinstance(unclean, sqlparse.sql.Parenthesis):
            clean_sql = clean_sql + (hash_and_salt(unclean, 0))
            continue
        elif isinstance(unclean, sqlparse.sql.Where):
            clean_sql = clean_sql + (hash_and_salt(unclean, 1))
            continue
        elif isinstance(unclean, sqlparse.sql.Comparison):
            clean_sql = clean_sql + (hash_and_salt(unclean, 2))
            continue
        elif isinstance(unclean, sqlparse.sql.IdentifierList):
            accum = ""
            for bla in unclean:
                bla = str(bla)
                if bla in ["=", "<", ">"]:
                    clean_sql = clean_sql + (hash_and_salt(unclean, 0))
                    # we don't want to potentially add the rest
                    break
                else:
                    # might be the case that the other identifiers down the road
                    # are assigned to, and we want to parse those out
                    accum = accum + bla
            clean_sql = clean_sql + accum
            continue
        else:
            # Handle the update case
            clean_sql = clean_sql + str(unclean)
            continue
    return clean_sql


def brutal_anonymize(sql):
    # A brutal version of anonymize(sql)
    # Faltten the sqlparse results, check for all the strings, and then salt
    # the non-dates and non-numerics-only ones.
    to_clean = None
    # If we can't parse it, then it is not of the form we want
    # e.g it is just a Quit statement or something
    try:
        to_clean = sqlparse.parse(sql)[0]
    except:
        return sql

    token_list = to_clean.flatten()

    clean_sql = ""
    last_token = ""

    for token in token_list:
        token_str = str(token)
        if ((token_str[0] != '\'' or token_str[-1] != '\'')
            and (token_str[0] != '\"' or token_str[-1] != '\"')):
            clean_sql = clean_sql + token_str
        else:
            if is_date_or_digits(token_str):
                clean_sql = clean_sql + token_str
            # escape anonymization after a 'LIKE' keyword when the string is not complete
            elif (last_token.upper().find('LIKE') != -1 and
                    (token_str.find('%') != -1 or token_str.find('_') != -1)):
                clean_sql = clean_sql + token_str
            else:
                data_length = str(len(token_str) - 2)
                cleaned = hashlib.md5(SALT +
                                      token_str.encode("utf-8")).hexdigest()
                clean_sql = clean_sql + " '" + \
                    (data_length + "\\" + str(cleaned)) + "'"

        if token_str not in string.whitespace:
            last_token = token_str

    return clean_sql


def process_query(query, type_):
    global OUTPUT
    """For the given query list, process the SQL string and anonymize it"""
    if type_ == "postgresql":
        try:
            # remove the double "EST" or "EDT"
            query = query[:1] + query[2:4] + query[5:]
            if ANONYMIZE == True:
                query[3] = brutal_anonymize(query[3])
                # HACK
                for regex in CLEAN_CMDS:
                    query[3] = regex.sub(r"\1 ", query[3])

                query[4] = brutal_anonymize(query[4])

            OUTPUT.writerow(query)
            return
        except:
            LOG.error("Failed to handle query '%s'" % query[-1])
            raise

    try:
        if not (query[-1].startswith("SET") or query[-1].startswith("SHOW")):
            # Depending on the version, the CMD might be the 2nd or the 3rd group
            if (query[3] == "Query" or query[2] == 'Query') and ANONYMIZE == True:

                query[-1] = brutal_anonymize(query[-1])
                # HACK
                for regex in CLEAN_CMDS:
                    query[-1] = regex.sub(r"\1 ", query[-1])
            OUTPUT.writerow(query)
    except:
        LOG.error("Failed to handle query '%s'" % query[-1])
        raise
# DEF


# ==============================================
# main
# ==============================================
if __name__ == '__main__':
    aparser = argparse.ArgumentParser(description='Log Anonymizer')
    aparser.add_argument('input', help='Input file')
    aparser.add_argument('--salt', default=None, metavar='H', help='Anonymization hash salt')
    aparser.add_argument('--no-anonymize', default=False, action='store_true', help='Disable anonymization')
    aparser.add_argument('--version', default='5.7', help='MySQL version')
    aparser.add_argument('--type', default='mysql', help='relational dbDesktop type')
    args = vars(aparser.parse_args())

    #pprint(args)
    if args['version'] == '5.5':
        LOG_REGEX = LOG_REGEX_5_5
    # Disable anonymization
    if 'no_anonymize' in args:
        ANONYMIZE = (args['no_anonymize'] == False)
    else:
        ANONYMIZE = True

    # Set hash salt
    if 'salt' in args and args['salt']:
        SALT = args['salt']
    else:
        SALT = "I fucking hate anonymizing queries"
    SALT = SALT.encode('utf-8')

    #pprint(args)
    #LOG.error("Anonymization is %s" % ("ENABLED" if ANONYMIZE else "DISABLED"))
    #LOG.error("Hash Salt: \"%s\"" % SALT)
    #sys.exit(1)

    files = []
    inputFile = args['input']
    if args['input'].lower().endswith(".gz"):
        f = gzip.open(args['input'], mode='rt',
                      encoding='utf-8', errors='replace')
        files.append(f)
    elif args['input'].lower().endswith(".zip"):
        zf = zipfile.ZipFile(args['input'], 'r')
        names = zf.namelist()
        for name in names:
            f = zf.open(name)
            files.append(f)
    else:
        f = open(args['input'], mode='r', encoding='utf-8', errors='replace')
        files.append(f)
    assert len(files) > 0

    # Work through them to clean them out just
    # like Taco Bell running through my intestinal tract...
    last_query = None
    last_time = None
    cnt = 0
    line = None
    if 'type' in args:
        type_ = args['type']

    try:
        for f in files:
            for line in f:
                if type_ == "postgresql":
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")

                    m = LOG_REGEX_POSTGRESQL.match(line)
                    if m:
                        if last_query is not None:
                            n = LOG_REGEX_POSTGRESQL.match(last_query)
                            if n:
                                #print("First", last_query)
                                process_query(list(n.groups()), type_)

                        last_query = line
                    else:
                        if POSTGRESQL_PREFIX_REGEX.match(line):
                            if last_query is not None:
                                m = LOG_REGEX_POSTGRESQL.match(last_query)
                                if m:
                                    #print("Second", last_query)
                                    process_query(list(m.groups()), type_)
                                else:
                                    last_query = None
                            else:
                                last_query = line.strip()
                        else:
                            if last_query is not None:
                                last_query += " " + line.strip()
                else:
                    m = LOG_REGEX.match(line)
                    if m:
                        if last_query is not None:
                            if last_query[0] is None:
                                last_query[0] = last_time
                            else:
                                last_time = last_query[0]
                            # Process this mofo
                            process_query(last_query, type_)
                            last_query = None
                        # IF
                        last_query = list(m.groups())
                    else:
                        # assert not (line.startswith("2016") or line.startswith("2017"))
                        if last_query is not None:
                            last_query[-1] += " " + line.strip()
                # IF
                cnt += 1
            # FOR
        # FOR
    except:
        LOG.error("Unexpected problem on line %d\n%s" % (cnt, line))
        raise
# MAIN
