#!/usr/bin/env python3

import sys
import psycopg2
import mysql.connector
import argparse
import random
import re
import csv
import re
import string
import time
import threading
import csv
import queue
import json
import multiprocessing
from multiprocessing import Process, Pool
from datetime import datetime
from datetime import timedelta
from getpass import getpass
from tqdm import tqdm

# Import Simulator
sys.path.append("../planner-simulator")
#from simulation import Simulator
from planner_simulator import Simulator

'''
Database Simulator and Trace Replayer
This is a database simulator that supports MySQL and PostgreSQL. The simulator is
run from the command line ("python3 workload-simulator.py -h" for usability). If the
database that the user specifies does not exist, then the generator can generate
it with a provided schema, and fill it with some fake data (see --rows). It
removes all the secondary indexes in the database, and only keeps the primary
indexes. After, it replays a trace provided by the user. At each fixed interval,
it creates a new index based on an index suggestion algorithm with the
forecasting result at that time.
There are three alternative modes used for comparison in the experiments:
    (1) create all the indexes before query replay (see STATIC_SUGGEST);
    (2) use the EXPLAIN SQL command to analyze how many queries are using
    indexes instead of actually executing the queries (see EXPLAIN);
    (3) use the clustering result with logical features instead of arrival rate
    features (see LOGICAL).
If one wants to assure that the traces from the queries actually execute and
return data, the simulator can insert data from the queries themselves prior to
executing the queries.
'''

#########################
## Workload Parameters ##
#########################

MODEL = "ensemble"

PROJECTS = {
    "tiramisu": {
        "name": "tiramisu",
        "db": "postgresql",
        "db_name": "tiramisu",
        "schema": "../mysql/combinedTiramisuSchema.sql",
        "workload": "tiramisu-out.log",
        "original_dir": "../time-series-clustering/tiramisu-combined-results",
        "predict_dirs": ["simulatorFiles/tiramisu-online-prediction/%s-60" % MODEL,
            "simulatorFiles/tiramisu-online-prediction/%s-1440" % MODEL,
            "simulatorFiles/tiramisu-online-prediction/%s-10080" % MODEL],
        "cluster_assignment": "simulatorFiles/cluster-coverage/tiramisu-assignments.pickle",
        "cluster_coverage": "simulatorFiles/cluster-coverage/tiramisu-coverage.pickle",
        "cluster_num": 3,
        "predict_aggregate": 60,
        "trace_aggregate": 10,
        "index_duration": 5,
        "warmup_period": 40,
        "output_file": "tiramisu-result.csv",

        "predict_dirs_logical": [
            "../../databases/prediction-logical-result/tiramisu/0.8/agg-60/horizon-60/%s" % MODEL,
            "../../databases/prediction-logical-result/tiramisu/0.8/agg-60/horizon-1440/%s" % MODEL,
            "../../databases/prediction-logical-result/tiramisu/0.8/agg-60/horizon-10080/%s" % MODEL,
            ],
        "cluster_assignment_logical":
            "../time-series-clustering/online-logical-clustering-results/tiramisu-0.8-assignments.pickle",
        "cluster_coverage_logical":
            "../time-series-clustering/cluster-coverage-logical/tiramisu/0.8/coverage.pickle",
    },
    "admission": {
        "name": "admission",
        "db": "mysql",
        "db_name": "adm",
        'schema': "../mysql/gradAdmissions2009New.sql",
        "workload": "admission-out.log",
        "original_dir": "simulatorFiles/combined-results",
        "predict_dirs": ["simulatorFiles/admission-online-prediction/%s-60" % MODEL,
            "simulatorFiles/admission-online-prediction/%s-1440" % MODEL,
            "simulatorFiles/admission-online-prediction/%s-10080" % MODEL],
        "cluster_assignment": "simulatorFiles/cluster-coverage/admission-assignments.pickle",
        "cluster_coverage": "simulatorFiles/cluster-coverage/admission-coverage.pickle",
        "cluster_num": 3,
        "predict_aggregate": 60, # Seconds
        "trace_aggregate": 10, # Seconds
        "index_duration": 5, # Seconds
        "warmup_period": 40, # Seconds
        "output_file": "admission-result.csv",

        "predict_dirs_logical": [
            "../../databases/prediction-logical-result/admission/0.2/agg-60/horizon-60/%s" % MODEL,
            "../../databases/prediction-logical-result/admission/0.2/agg-60/horizon-1440/%s" % MODEL,
            "../../databases/prediction-logical-result/admission/0.2/agg-60/horizon-10080/%s" % MODEL,
            ],
        "cluster_assignment_logical":
            "../time-series-clustering/online-logical-clustering-results/admission-0.2-assignments.pickle",
        "cluster_coverage_logical":
            "../time-series-clustering/cluster-coverage-logical/admission/0.2/coverage.pickle",
    },
    "oli": {
        "name": "oli",
        "db": "mysql",
        "db_name": "oli",
        'schema': "../mysql/olischema.sql",
        "workload": "oli-out.log",
        "original_dir": "../time-series-clustering/oli-combined-results",
        "predict_dirs": ["simulatorFiles/oli-online-prediction/%s-60" % MODEL,
            "simulatorFiles/oli-online-prediction/%s-1440" % MODEL,
            "simulatorFiles/oli-online-prediction/%s-10080" % MODEL],
        "cluster_assignment": "simulatorFiles/cluster-coverage/oli-assignments.pickle",
        "cluster_coverage": "simulatorFiles/cluster-coverage/oli-coverage.pickle",
        "cluster_num": 3,
        "predict_aggregate": 60,
        "trace_aggregate": 10,
        "output_file": "oli-result-all-index.csv"
    }
}


#########################
## Database Parameters ##
#########################

DATABASES = {
    "mysql": {
        "error": mysql.connector.Error,
        "create_sql": "CREATE DATABASE {} DEFAULT CHARACTER SET 'utf8'",
        "index_sql": """SELECT DISTINCT INDEX_NAME, TABLE_NAME FROM INFORMATION_SCHEMA.STATISTICS
                        where table_schema='{}';""",
        "index_set_sql": """SELECT DISTINCT TABLE_NAME, COLUMN_NAME FROM
                            INFORMATION_SCHEMA.STATISTICS where table_schema='{}';""",
        "column_set_sql": """SELECT table_name, column_name, DATA_TYPE FROM
                            INFORMATION_SCHEMA.COLUMNS where table_SCHEMA = 'adm';""",
        "drop_sql": "DROP INDEX `{}` ON {};",
        'pwd': "pass",
        "explain_prefix": "explain format = json ",
    },
    "postgresql": {
        "error": psycopg2.Error,
        "create_sql": "CREATE DATABASE {}",
        "index_sql": """select indexrelid::REGCLASS, indrelid::REGCLASS from pg_index where
                        indexrelid::REGCLASS::VARCHAR not like 'pg_%';""",
        "index_set_sql": """select
                         ix.indrelid::REGCLASS,
                         a.attname
                        from
                         pg_index ix,
                         pg_attribute a
                        where
                         a.attrelid = ix.indrelid
                         and a.attnum = ANY(ix.indkey)
                         and ix.indexrelid::REGCLASS::VARCHAR not like 'pg_%';""",
        "column_set_sql": """select table_schema::text||'.'||table_name::text,
                         column_name, data_type from information_schema.columns
                         where table_schema not in ('pg_catalog', 'information_schema');""",
        "drop_sql": "DROP INDEX {};",
        "user": "admin",
        'pwd': "password",
        "host": "localhost",
        "explain_prefix": "explain (format json) ",
    },
}

################
## Arg Parser ##
################

parser = argparse.ArgumentParser(description='Populate fake database and simulate workload')
parser.add_argument("-u", "--username", help="MySQL Log-in username (Default: root)",
                    type=str)
parser.add_argument("--schema", help="SQL schema to populate database")
parser.add_argument("--rows", help="Number of rows to populate database tables with (Default: 500)",
                        type=int)
parser.add_argument("--workload", help="File containing newline separated queries")
parser.add_argument("--host", help="Host for MySQL (Default: 127.0.0.1)")
parser.add_argument("--port", help="Port for MySQL (Default: 3306)")
parser.add_argument("--num_queries", type=int, help="Maximum number of queries to execute")
parser.add_argument('--logical', action='store_true', help="Whether to use the logical clustering result")
parser.add_argument('--static_suggest', action='store_true', help="Whether to suggest all the"
        "indexes statically")
parser.add_argument('--project', choices=PROJECTS.keys(), help='Data source type')


args = parser.parse_args()

#################
## Global Vars ##
#################
# Whether use explain to analyze how many queries are using indexes
EXPLAIN = False

# Whether use the logical clustering result
LOGICAL = args.logical

# Whether to generate all the indexes during warmup period based on the entire workload
STATIC_SUGGEST = args.static_suggest


HOST = args.host or "127.0.0.1"
PORT = args.port or 3306
ROWS = args.rows or 500
USERNAME = args.username or "root"
NUM_ITERS = 1
BATCH_LENGTH = 1  # Seconds
DATA_PROCESSES = 8
NUM_QUERIES = args.num_queries
CONFIG = PROJECTS[args.project]
CONFIG["schema"] = args.schema or CONFIG["schema"]
CONFIG["workload"] = args.workload or CONFIG["workload"]
CONFIG["num_process"] = 24
if LOGICAL:
    CONFIG['predict_dirs'] = CONFIG['predict_dirs_logical']
    CONFIG['cluster_assignment'] = CONFIG['cluster_assignment_logical']
    CONFIG['cluster_coverage'] = CONFIG['cluster_coverage_logical']

# MySQL/PostgreSQL Data types
INT_TYPES = ["BIT", "TINYINT", "SMALLINT", "MEDIUMINT", "INT", "INTEGER", "BIGINT"]
STRING_TYPES = ["CHAR", "VARCHAR", "BINARY", "CHARACTER VARYING"]
TEXT_TYPES = ["TINYTEXT", "TEXT", "MEDIUMTEXT", "LONGTEXT", "BLOB", "MEDIUMBLOB", "LONGBLOB", "BYTEA"]
FLOAT_TYPES = ["DECIMAL", "FLOAT", "REAL", "DOUBLE", "DOUBLE PRECISION"]
TIME_TYPES = ["TIMESTAMP", "DATETIME", "DATE", "TIME", "TIMESTAMP WITHOUT TIME ZONE",
        "TIME WITHOUT TIME ZONE", "TIMESTAMP WITH TIME ZONE"]
BOOL_TYPES = ["BOOLEAN"]
LOCATION_TYPES = ["POINT", "POLYGON", "BOX"]

# Regex
QUERY_DATA_RE = re.compile(r'(\S+\s*[=<>]+\s*\S+)')
HASH_SPLIT_RE = r'\\+'




######################
## Optimizer Thread ##
######################

class OptimizerThread(threading.Thread):
    def __init__(self, dbconfig, timestamp,duration,index_set, conn, cur, writer, row):
        threading.Thread.__init__(self)
        self.error = dbconfig['error']
        self.timestamp = timestamp
        self.duration = duration
        self.index_set = index_set
        self.conn = conn
        self.cur = cur
        self.writer = writer
        self.row = row

    def run(self):
        # Create index
        try:
            pair = SimulatorObject.SuggestIndex(self.timestamp,
                                                   self.duration,
                                                   self.index_set)
            if pair != None:
                index = pair[1]
                table = pair[0]
                self.row.append(table + "." + index)
                self.writer.writerow(self.row)

                print("\n#=============================")
                print("Creating index {} on {}".format(index,table))
                print("#=============================")
                index_table = table.replace('.', '_')
                query = "CREATE INDEX {} ON {}({});".format("{}_{}_idx".format(index_table, index), table, index)
                self.cur.execute(query)
                self.conn.commit()
                print("#=============================")
                print("Created index {} on {}".format(index,table))
                print("#=============================")
            else:
                self.row.append("None")
                self.writer.writerow(self.row)

        except mysql.connector.Error as err:
            print("\n#=============================")
            print("ERROR: %s", err)
            print("QUERY:%s", query)
            print("\n#=============================")

        return


###########################
## Scan Statistics ##
###########################

class ScanStats():
    def __init__(self):
        # index scan statistics
        self.seq_q = 0
        self.index_q = 0
        self.primary_q = 0
        self.foreign_q = 0
        self.secondary_q = 0

    def __str__(self):
        s = ""
        s += "seq: " + str(self.seq_q)
        s += " idx: " + str(self.index_q)
        s += " primary: " + str(self.primary_q)
        s += " foreign: " + str(self.foreign_q)
        s += " secondary: " + str(self.secondary_q)
        return s

    def MergeStats(self, stats):
        self.seq_q += stats.seq_q
        self.index_q += stats.index_q
        self.primary_q += stats.primary_q
        self.foreign_q += stats.foreign_q
        self.secondary_q += stats.secondary_q
        return

    def GetStats(self):
        return [self.seq_q, self.index_q, self.primary_q, self.foreign_q, self.secondary_q]




###############
## Functions ##
###############

# Flatten a jason object
def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                #flatten(x[a], name + a + '_')
                flatten(x[a], a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                #flatten(a, name + str(i) + '_')
                flatten(a, name + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

# Close database connection
def exitDatabase(cnx, cursor):
    #print("Closing database connection")
    cursor.close()
    cnx.close()
    exit(1)


# Create the required database
def createDatabase(config, dbconfig, cnx, cursor):
    try:
        cursor.execute(
            dbconfig['create_sql'].format(config['db_name']))
    except dbconfig['error'] as err:
        print("ERROR: Failed creating database: {}".format(err))
        exitDatabase(cnx, cursor)

def openMySQLConnection(dbconfig):
    try:
        cnx = mysql.connector.connect(user=USERNAME, password=dbconfig['pwd'],
                                        host=HOST, port=PORT,
                                        buffered=True)
        #print("Sucessfully connected to MySQL server")
        return cnx
    except:
        print("ERROR: Failed to connect to MySQL server")
        exit(1)

# Connects to DB if already exists, else it creates and populates it
def connectToMySQLDatabase(config, dbconfig, cnx, cursor):
    db_name = config['db_name']

    try:
        cnx.database = db_name
        #print("Sucessfuly found database '{}'".format(db_name))

    # Database doesnt exist, needs to be created
    except mysql.connector.Error as err:
        if err.errno == mysql.connector.errorcode.ER_BAD_DB_ERROR:
            user_input = input("Database '{}' not found. Would you like to create it (y/n):".format(db_name))
            if user_input.lower() == "y" or user_input.lower() == "yes":
                createDatabase(config, dbconfig, cnx, cursor)
                cnx.database = db_name
            else:
                exitDatabase(cnx, cursor)
        else:
            print(err)
            exitDatabase(cnx, cursor)

        # Populate Database with schema
        if config["schema"]:
            cursor = cnx.cursor()
            createTables(config, dbconfig, cursor)
        else:
            print("ERROR: Please provide schema")
            cursor.execute("DROP DATABASE {};".format(db_name)) # delete empty database
            exitDatabase(cnx, cursor)


        # Generate the fake data
        generateDataMultiProcess(config, dbconfig, cnx, cursor)

# Connects to DB if already exists, else it creates and populates it
def connectToPostgresDatabase(config, dbconfig):
    db_name = config['db_name']

    try:
        cnx = psycopg2.connect(host=dbconfig['host'], dbname=db_name, user=dbconfig['user'],
                password=dbconfig['pwd'])
        print("Sucessfuly found database '{}'".format(db_name))
    # Database doesnt exist, needs to be created
    except dbconfig['error'] as err:
        try:
            cnx = psycopg2.connect(host=dbconfig['host'], dbname='postgres', user=dbconfig['user'],
                    password=dbconfig['pwd'])
            cnx.autocommit = True
            cursor = cnx.cursor()
            createDatabase(config, dbconfig, cnx, cursor)
            print("Created database {}".format(db_name))

        except dbconfig['error'] as err:
            print(err)
            print("ERROR: Failed to connect to PostgreSQL server")
            exit(1)

        cnx = psycopg2.connect(host=dbconfig['host'], dbname=db_name, user=dbconfig['user'],
                password=dbconfig['pwd'])

        cnx.autocommit = True
        cursor = cnx.cursor()
        createTables(config, dbconfig, cursor)

        # Generate the fake data
        generateDataMultiProcess(config, dbconfig, cnx, cursor)
        cursor.close()

    cnx.autocommit = True
    cursor = cnx.cursor()

    return cnx, cursor

def connectToDatabase(config, dbconfig):
    if config['db'] == 'mysql':
        cnx = openMySQLConnection(dbconfig)
        cnx.autocommit = True
        cursor = cnx.cursor()
        connectToMySQLDatabase(config, dbconfig, cnx, cursor)

    if config['db'] == 'postgresql':
        cnx, cursor = connectToPostgresDatabase(config, dbconfig)

    return cnx, cursor

# Populates a database with tables using the given schema
def createTables(config, dbconfig, cursor):
    schema = config['schema']
    if config['db'] == 'mysql':
        cursor.execute("SET sql_mode = '';")

    print("Populating '{}' with tables...".format(config['db_name']), end="")
    # Split by query
    schema = open(schema, 'r').read().split(";\n\n")
    for i in tqdm(range(len(schema)), desc="Creating Tables", ncols=100):
    #for i in range(len(schema)):
        # theres probably a better way to do this using regex but its 11:51pm and I hate regex
        query = schema[i] + ";"  # add back ';' lost in splitting
        if len(query) < 4:
            continue
        if "USE " in query:
            continue
        if "DATABASE" in query:
            continue
        try:
            print(query)
            print("-" * 100)
            if config['db'] == 'mysql':
                for q in query.split(";\n"):
                    if "Final view" in q:
                        print("Done!")
                        return
                    print(q)
                    cursor.execute(q)
            else:
                cursor.execute(query)
        except dbconfig['error'] as err:
            print(err)
            exit()
            continue
    print("Done!")



def getTables(config, dbconfig, cursor):
    if config['db'] == "mysql":
        cursor.execute("SHOW TABLES;")
        tables = [x[0] for x in cursor]
    
    if config['db'] == 'postgresql':
        cursor.execute("SELECT * FROM pg_tables t WHERE t.tableowner ='{}'".format(dbconfig['user']))
        tables = [x[0] + "." + x[1] for x in cursor]

    return tables

def getColumns(cursor, table_name, table_schema):
    # Get columns
    cursor.execute("""SELECT COLUMN_NAME FROM
                    INFORMATION_SCHEMA.COLUMNS WHERE
                    TABLE_NAME='{}' AND TABLE_SCHEMA='{}';""".format(table_name, table_schema))
    columns = [x[0] for x in cursor] # columns for table
    return columns


def getColumnTypes(cursor, table_name, table_schema):
    cursor.execute("""SELECT DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE table_name = '{}' AND TABLE_SCHEMA='{}';""".format(table_name,
                        table_schema))
    types = [x[0] for x in cursor]
    return types

def getColumnTypeSizes(cursor, table_name, table_schema):
    cursor.execute("""SELECT CHARACTER_MAXIMUM_LENGTH FROM
                      information_schema.columns WHERE TABLE_NAME='{}'
                      AND TABLE_SCHEMA='{}'; """.format(table_name, table_schema))
    result = cursor.fetchall()
    sizes = []
    for x in result:
        x = x[0]
        if x == None or x == "NULL":
            sizes.append(None)
        else:
            try:
                sizes.append(int(x))
            except:
                sizes.append(None)
    return sizes

def getData(col_type, size):
    col_type = col_type.upper()
    # INT TYPES
    if col_type == "TINYINT" or col_type == "BIT":
        return (random.randint(0,1))
    elif col_type == "SMALLINT":
        return (random.randint(0, (2**16)-1))
    elif col_type == "MEDIUMINT":
        return (random.randint(0, (2**24)-1))
    elif col_type in ["INT", "INTEGER", "BIGINT", "OID"]:
        return (random.randint(0, (2**24)-1))
    elif col_type in INT_TYPES and size == None:
        return 1
    
    # Boolean Types
    elif col_type in BOOL_TYPES:
        return True if random.randint(0,1) > 0 else False

    # Boolean Types
    elif col_type == "POINT":
        return "(" + str(random.randint(0, (2**24)-1)) + ", " + str(random.randint(0,
            (2**16)-1)) + ")"

    elif col_type in LOCATION_TYPES:
        return "NULL"

    # Random Types
    elif col_type in TEXT_TYPES:
        return ("FakeNews")
    elif col_type == "ENUM":
        return ""
    elif col_type == "YEAR":
        return 2017

    # String Types
    elif col_type in STRING_TYPES:
        if size != None:
            return (''.join(random.choice(string.ascii_lowercase) for x in range(size)))
        else:
            return (''.join(random.choice(string.ascii_lowercase) for x in range(10)))

    elif col_type in FLOAT_TYPES:
        return random.random()
    elif col_type in TIME_TYPES:
        return datetime.now()

    else:
        return None

def getEnumData(config, cursor, table_schema, table_name, col):
    if config['db'] == 'mysql':
        cursor.execute("""SELECT COLUMN_TYPE FROM
                        INFORMATION_SCHEMA.COLUMNS WHERE
                        TABLE_NAME='{}' AND TABLE_SCHEMA='{}'
                        AND COLUMN_NAME='{}';""".format(table_name, table_schema, col))
        result = cursor.fetchone()[0]
        value = result.split("'")[1]

    if config['db'] == 'postgresql':
        cursor.execute("""SELECT UDT_NAME FROM
                        INFORMATION_SCHEMA.COLUMNS WHERE
                        TABLE_NAME='{}' AND TABLE_SCHEMA='{}'
                        AND COLUMN_NAME='{}';""".format(table_name, table_schema, col))
        udt_name = cursor.fetchone()[0]
        cursor.execute("""SELECT ENUM_RANGE(NULL::{}.{})""".format(table_schema, udt_name))
        values = cursor.fetchone()[0]
        value = values[1:-1].split(",")[0]

    return value

def generateDataMultiProcess(config, dbconfig, cnx, cursor):
    tables = getTables(config, dbconfig, cursor)
    process_size = len(tables) // DATA_PROCESSES

    proc = []
    for i in range(DATA_PROCESSES):
        # Process log
        p = Process(target = generateData, args = (config, dbconfig, tables[i * process_size:
                (i + 1) * process_size]))
        p.start()
        proc.append(p)

    for p in proc:
        p.join()



# Generates data for a given schema
# NOTE: to deal with foreign keys, I use the same data for the ith row in each
#       table

def generateData(config, dbconfig, tables):
    print("Start loading process...")

    cnx, cursor = connectToDatabase(config, dbconfig)

    # Insert Fake data table by table
    if config['db_name'] == 'mysql':
        cursor.execute("SET FOREIGN_KEY_CHECKS=0;") # this is a hack lol
        cnx.commit()

    for i in tqdm(range(len(tables)), desc="Generating Fake Data", ncols=100):
        table = tables[i]
        if config['db'] == 'mysql':
            table_name = table
            table_schema = config['db_name']

        if config['db'] == 'postgresql':
            table_split = table.split(".")
            table_name = table_split[1]
            table_schema = table_split[0]

        columns = getColumns(cursor, table_name, table_schema) # columns for table
        types = getColumnTypes(cursor, table_name, table_schema)
        col_sizes = getColumnTypeSizes(cursor, table_name, table_schema)

        INSERT_STRING = "INSERT INTO {} VALUES (".format(table)
        INSERT_STRING += "%s," * len(columns)
        INSERT_STRING = INSERT_STRING[:-1] + ");"
        #ALL_DATA = []
        # Insert ROWS number of rows of fake data
        for i in range(ROWS):
            # Get fake data for 1 row
            INSERT_DATA = []
            for j in range(len(columns)):
                col_name = columns[j]
                col_type = types[j].upper()
                if col_type in ["ENUM", "SET", "USER-DEFINED"]: # chose a random one for SET data
                    INSERT_DATA.append(getEnumData(config, cursor, table_schema, table_name, col_name))
                else:
                    size = col_sizes[j]
                    data = getData(col_type, size)
                    if data != None:
                        if data == "NULL":
                            INSERT_DATA.append(None)
                        else:
                            INSERT_DATA.append(data)

                    else:
                        print("Column: '{} | Table: '{}' | Type: {} Not supported".format(col_name, table, col_type))
                        print("If this prints yell at Gus to cover all SQL types :)")
                        exitDatabase(cnx, cursor)
            #ALL_DATA.append(tuple(INSERT_DATA))
            try:
                cursor.execute(INSERT_STRING, INSERT_DATA)
            except dbconfig['error'] as err:
                #print("\n")
                #print(INSERT_STRING % tuple(ALL_DATA[0]))
                #print(err)
                #print("\n")
                continue
        cnx.commit()

    if config['db_name'] == 'mysql':
        cursor.execute("SET FOREIGN_KEY_CHECKS=1;") # not sorry
    print("Done!")

    exitDatabase(cnx, cursor)


def getTableFromQuery(query):
    query = query.split()
    try:
        table = query[query.index("from") + 1] # should be keyword follower from
        if table[-1] == ";":
            table = table[:-1]
        return table
    except:
        return ""

def getQueryData(query, query_dict):
    query = re.split(QUERY_DATA_RE, query)
    for sub in query:
        if sub.count("=") == 1:
            sub = sub.split("=")
        elif sub.count(">") == 1:
            sub = sub.split(">")
        elif sub.count("<") == 1:
            sub = sub.split("<")
        else:
            continue
        # Data field
        if len(sub) != 2: # ['column_name', "##//hash"]
            continue
        for i in range(len(sub)):
            sub[i] = sub[i].lstrip().rstrip() #strip whitspace

        # remove possible ; in the end
        if sub[1][-1] == ";":
            sub[1] = sub[1][:-1]

        # Get value
        # if its a hash of the form x\\abcdefg..., the value is the first
        # x characters of abcdefg...
        if sub[1].count("\\") > 1:
            try:
                split_hash = re.split(HASH_SPLIT_RE, sub[1])
                if len(split_hash) != 2:
                    continue
                query_dict[sub[0]] = split_hash[1][:int(split_hash[0])]
            except:
                continue
        else:
            query_dict[sub[0]] = sub[1]


# Returns true if it was able to insert query
def insertData(cursor, query):
    global cnx
    table = getTableFromQuery(query)
    if table == "": #couldnt find table
        return False
    query_data = dict()
    getQueryData(query, query_data) # gets any data it can from the query
    columns = getColumns(cursor,table)
    columnTypes = getColumnTypes(cursor, table)
    columnSizes = getColumnTypeSizes(cursor, table)
    INSERT_DATA = []
    for i in range(len(columns)):
        column = columns[i].lower()
        if column in query_data:
            INSERT_DATA.append(query_data[column])
        else:
            data = getData(columnTypes[i], columnSizes[i])
            INSERT_DATA.append(data)

    # Insert the data
    INSERT_STRING = "INSERT INTO {} VALUES (".format(table)
    INSERT_STRING += "%s," * len(columns)
    INSERT_STRING = INSERT_STRING[:-1] + ");"
    try:
        cursor.execute(INSERT_STRING, tuple(INSERT_DATA))
    except mysql.connector.Error as err:
        #print("\nERROR: {}".format(err))
        return False

    return True

def preprocessQuery(query):
    oldquery = query
    query = re.split(r"'(\d+\\+.*)'", query)
    for i in range(len(query)):
        sub = query[i]
        if sub.count("\\") > 0:
            try:
                split_hash = re.split(HASH_SPLIT_RE, sub)
                if len(split_hash) != 2:
                    continue
                query[i] = "'" + split_hash[1][:int(split_hash[0])] + "'"
            except:
                continue
    query = ''.join(query)
    return query


def unpackLine(line):
    line_split = line.split(",")
    time_stamp = datetime.strptime(line_split[0], '%Y-%m-%d %H:%M:%S')
    time_stamp = time_stamp.replace(second=0, microsecond=0)
    conn_id = line_split[1]
    query = ",".join(line_split[2:])
    return (time_stamp, conn_id, query)

def dropIndexes(config, dbconfig, cursor):
    query = dbconfig['index_sql'].format(config['db_name'])
    cursor.execute(query)
    cnx.commit()
    indexes = cursor.fetchall()
    for index in indexes:
        index_name = index[0]
        table_name = index[1]
        if index_name.upper() != 'PRIMARY' and ("fk_" not in index_name) and ("pkey" not in
                index_name):
            #continue
            query = dbconfig['drop_sql'].format(index_name, table_name)
        else:
            # comment out to drop primary
            continue
            query = dbconfig['drop_sql'].format(index_name, table_name)

        try:
            cursor.execute(query)
            print("Successfully drop index: ", query)
        except dbconfig['error'] as err:
            print(query)
            print(err)
            continue

def getBatch(timestamp, file_lines):
    filelen = len(file_lines)

    end_index = 0
    while (end_index < filelen and file_lines[end_index][0] <= timestamp):
        end_index += 1

    return file_lines[:end_index], file_lines[end_index:]

def getIndexSet(config, dbconfig, cursor):
    query = dbconfig['index_set_sql'].format(config['db_name'])
    cursor.execute(query)
    cnx.commit()
    indexes = cursor.fetchall()
    #print(indexes)
    if indexes == None:
        return set()
    else:
        return set(indexes)


########################################
## Process Query with Multi-processes ##
########################################
def initWorker(config, dbconfig, function):
    function.cnx, function.cursor = connectToDatabase(config, dbconfig)

def executeQueryMultiProcess(config, dbconfig, pool, batch):
    batch_dict = dict()
    for i in range(config['num_process']):
        batch_dict[i] = list()

    # round robin queries to 10 bins
    cnt = 0
    for time_stamp, conn, query in batch:
        batch_dict[cnt].append((time_stamp, query))

        cnt += 1
        if cnt == config['num_process']:
            cnt = 0

    proc = []
    result_dict = dict()
    print("Number of conns: {}".format(len(batch_dict)))
    for conn, batch in batch_dict.items():
        print("Connection {} with {} queries.".format(conn, len(batch)))
        # Execute queries
        res = pool.apply_async(executeQuery, (config, dbconfig, batch))
        result_dict[conn] = res

    exec_q = 0
    curr_q = 0
    latency = []
    scan_stats = ScanStats()
    res = None
    for conn, result in result_dict.items():
        try:
            res = result.get(timeout = 20)
        except multiprocessing.TimeoutError as err:
            print("Connection {} didn't finish execution.".format(conn))
            continue

        print("Connection {} with results {} {} {}".format(conn, res[0], res[1], len(batch_dict[conn])))
        exec_q += res[0]
        curr_q += res[1]
        latency = latency + res[2]
        scan_stats.MergeStats(res[3])

    return (exec_q, curr_q, latency, scan_stats)

def executeQuery(config, dbconfig, batch):
    print("Start executing process...")

    cnx, cursor = executeQuery.cnx, executeQuery.cursor
    res = None

    # execution statistics
    curr_q = -1
    exec_q = 0

    scan_stats = ScanStats()

    latency = []
    timeout = time.time() + BATCH_LENGTH  # 1 Second
    while ((time.time() < timeout) and (curr_q < len(batch) - 1)):
        try:
            curr_q += 1
            start_time = time.time()
            if not EXPLAIN:
                cursor.execute(batch[curr_q][1])
            else:
                cursor.execute(dbconfig['explain_prefix'] + batch[curr_q][1])
                plan = cursor.fetchall()[0][0]
                if config['db'] == 'mysql':
                    plan = json.loads(plan)
                plan_dict = flatten_json(plan)

                use_primary = False
                use_foreign = False
                use_secondary = False
                for key, value in plan_dict.items():
                    if value == "no matching row in const table":
                        use_primary = True
                        break

                    if key in ["possible_keys_", 'key', "Index Name"]:
                        if "PRIMARY" in value or "pkey" in value:
                            use_primary = True
                        elif "fk_" in value or "fkey" in value:
                            use_foreign = True
                        else:
                            use_secondary = True

                if use_primary:
                    scan_stats.primary_q += 1
                elif use_foreign:
                    scan_stats.foreign_q += 1
                elif use_secondary:
                    scan_stats.secondary_q += 1
                else:
                    if scan_stats.seq_q < 10:
                        print(batch[curr_q][1])
                    scan_stats.seq_q += 1

            elapsed_time = time.time() - start_time
            latency.append(elapsed_time)
            exec_q += 1

            if EXPLAIN:
                scan_stats.index_q = exec_q - scan_stats.seq_q

            # update the result for this connection
            res = (exec_q, curr_q, latency, scan_stats)

        except dbconfig['error'] as err:
            #print("\n")
            #print(batch[curr_q][1])
            #print(err)
            #print("\n")
            #input("break")
            continue
    cnx.commit()

    return res


def runTrace(config, dbconfig, cnx, cursor):
    print("Start changing configurations...")
    if config['db'] == 'mysql':
        cursor.execute("SET FOREIGN_KEY_CHECKS=0;") # this is a hack, sorry
        cursor.execute("SET GLOBAL sql_mode ="
                "'ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,"
                "NO_ENGINE_SUBSTITUTION,ALLOW_INVALID_DATES';")
        cnx.commit()

    if (not config["workload"]):
        print("ERROR: Please provide workload")
        exitDatabase(cnx, cursor)
    print("Simulating workload...")

    # Open Trace and Store data
    file_lines = []
    with open(config["workload"], "r", encoding="UTF-8") as f:
        n = sum(1 for line in f)

        if NUM_QUERIES is not None:
            n = min(n, NUM_QUERIES)

        f.seek(0)
        for i in tqdm(range(n), desc="Processing Trace", ncols=100):
            line = f.readline()
            line = unpackLine(line)
            if line is not None:
                file_lines.append(line)
    f.close()

    # Create data from Trace
    #userInput = input("Create data from trace? (y/n) ")
    #if userInput.lower() == "y":
    #    for i in tqdm(range(len(file_lines)), desc="Creating Data from Trace", ncols=100):
    #        query = file_lines[i][1].lower()
    #        if "join" in query: # Skip joins
    #            continue
    #        insertData(cursor, query)
    #    cnx.commit()

    # Drop Indexs
    dropIndexes(config, dbconfig, cursor)

    # Run through trace in minute batches
    # Execute a minute worth of queries in one second
    MIN_TIMESTAMP = file_lines[0][0]
    MAX_TIMESTAMP = file_lines[-1][0]
    NUM_MINUTES = int(((MAX_TIMESTAMP - MIN_TIMESTAMP).total_seconds())/60)

    # Store Threads
    simThreads = []
    
    # Don't pollute the real output in explain mode
    if not EXPLAIN:
        result_file = "results-" + MODEL + "/" + config['output_file']
    else:
        result_file = "stats/explain-" + config['output_file']

    if LOGICAL:
        result_file = "logical-" + result_file

    if STATIC_SUGGEST:
        result_file = "static2-" + result_file
        config['index_duration'] = 2
        print(result_file)

    pool = Pool(processes = config['num_process'], initializer = initWorker,  initargs=(config, dbconfig, executeQuery))

    for i in range(NUM_ITERS):
        print("# ==========================")
        print("# Iteration {}".format(i))
        print("# ==========================")
        with open(result_file, "w+") as file:
            writer = csv.writer(file)
            latency = []
            total_scan_stats = ScanStats()
            index_cnt = 0
            print("Start Warming Up.\n")

            for j in range(0, NUM_MINUTES, config["trace_aggregate"]):
                curr_timestamp = MIN_TIMESTAMP + timedelta(minutes=j)
                batch, file_lines = getBatch(curr_timestamp +
                        timedelta(minutes=config["trace_aggregate"]), file_lines)

                #Store results
                row = []
                row.append(curr_timestamp)

                exec_q, curr_q, latest_latency, scan_stats = executeQueryMultiProcess(config,
                        dbconfig, pool, batch)

                latency = latency + latest_latency

                current_latency = sorted(latency[-3000:])
                percentile_index = (len(current_latency) // 100) + 1
                percentile_latency = current_latency[-percentile_index]

                if not EXPLAIN:
                    row.append(float(exec_q)/BATCH_LENGTH) # Add throughput
                    row.append(percentile_latency)
                else:
                    row += scan_stats.GetStats()

                # Print Confirmation
                print("Batch: {} {}mins".format(curr_timestamp, config['trace_aggregate']))
                print("Throughput: {} queries/sec".format(float(exec_q)/BATCH_LENGTH))
                print("99% Latency: {} sec/query".format(float(percentile_latency)))
                print("Executed: {}/{} queries ({}%)\n".format(curr_q,len(batch),
                    float(exec_q)/float(len(batch) + 1) * 100))

                if EXPLAIN:
                    print("current scan stats: " + str(scan_stats) + "\n")
                    print("total scan stats: " + str(total_scan_stats) + "\n")

                if j // config['trace_aggregate'] * BATCH_LENGTH < config['warmup_period']:
                    print("Continue warm up ...\n")
                    if STATIC_SUGGEST is False:
                        continue

                # Create at most 20 indexes
                if STATIC_SUGGEST and index_cnt >= 20:
                    config["index_duration"] = 1000000

                total_scan_stats.MergeStats(scan_stats)

                # Create Index every index_duration seconds
                if ((j // config['trace_aggregate']+ 1) % config["index_duration"] == 0):
                    # give beginning timestamp
                    start_timestamp = curr_timestamp - timedelta(minutes=(config["index_duration"]-1))

                    index_cnx, index_cursor = connectToDatabase(config, dbconfig)

                    t = OptimizerThread(dbconfig,
                                    timestamp=start_timestamp,
                                    duration=config["index_duration"] * config['trace_aggregate'],
                                    index_set=getIndexSet(config, dbconfig, cursor),
                                    conn=index_cnx,
                                    cur=index_cursor,
                                    writer=writer,
                                    row=row)
                    simThreads.append(t)
                    t.start()
                    if STATIC_SUGGEST:
                        t.join()
                    index_cnt += 1
                else:
                    row.append("None")
                    writer.writerow(row)
                    file.flush()


    if config['db'] == 'mysql':
        cursor.execute("SET FOREIGN_KEY_CHECKS=1;") # this is a hack, sorry
        cnx.commit()

    print("Done!")

def getColumnCardinality(config, dbconfig, cnx, cursor):
    cursor.execute(dbconfig['column_set_sql'])
    columns = cursor.fetchall()
    card_dict = {}
    for table, column, data_type in columns:
        data_type = data_type.upper()

        valid_types = INT_TYPES + FLOAT_TYPES + TIME_TYPES + ["VARCHAR", "CHARACTER VARYING"]

        card = 0
        if (data_type in valid_types) and (data_type != "BIT"):
            card = 1

        card_dict[(table, column)] = card

    cnx.commit()

    return card_dict

if __name__ == '__main__':
    config = CONFIG
    dbconfig = DATABASES[config['db']]

    cnx, cursor = connectToDatabase(config, dbconfig)

    column_card = getColumnCardinality(config, dbconfig, cnx, cursor)

    SimulatorObject = Simulator(config['schema'],
                                config['original_dir'],
                                config['predict_dirs'],
                                config['cluster_assignment'],
                                config['cluster_coverage'],
                                config['cluster_num'],
                                config['predict_aggregate'],
                                column_card,
                                STATIC_SUGGEST
                                )

    runTrace(config, dbconfig, cnx, cursor)
    exitDatabase(cnx, cursor)

