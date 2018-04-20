import sys
import csv
import gzip
import datetime
import re
import hashlib

# ==============================================
# PROJECT CONFIGURATIONS
# ==============================================

PROJECTS = {
    "tiramisu": {
        "name": "tiramisu",
        "files": "dbp*postgresql-*.anonymized.gz",
        "mysql": False,
        "query_index": 3,
        "param_index": 4,
        "conn_pos": 1,
        "time_stamp_format": "%Y-%m-%d %H:%M:%S",
        "schema": "../mysql/combinedTiramisuSchema.sql",
    },
    "admission": {
        "name": "admission",
        "files": "magneto.log.*.anonymized.gz",
        "mysql": True,
        "type_index": 3,
        "query_index": 4,
        "conn_pos": 2,
        "time_stamp_format": "%Y-%m-%d %H:%M:%S",
        'schema': "../mysql/gradAdmissions2009New.sql",
    },
    "oli": {
        "name": "oli",
        "files": "db*logfile*.anonymized.gz",
        "mysql": True,
        "type_index": 2,
        "query_index": 3,
        "conn_pos": 1,
        "time_stamp_format": "%y%m%d %H:%M:%S",
        'schema': "../mysql/olischema.sql",
    }
}

STATEMENTS = ['select', 'SELECT', 'INSERT', 'insert', 'UPDATE', 'update', 'delete', 'DELETE']
SALT = "I fucking hate anonymizing queries"
SALT = SALT.encode('utf-8')


def GetEnumDict(schema_file):
    enum_dict = dict()
    sql_schema = open(schema_file, 'r')
    for line in sql_schema:
        enum_match = re.search("(ENUM|enum)\((.*)\)", line)
        #print(line)
        if enum_match is not None:
            enums = enum_match.group(2)
            enums = re.split(", | |,", enums)

            for enum in enums:
                data_length = str(len(enum) - 2)
                cleaned = hashlib.md5(SALT +
                                      enum.encode("utf-8")).hexdigest()
                clean_enum = "'" + (data_length + "\\" + str(cleaned)) + "'"
                enum_dict[clean_enum] = enum

    return enum_dict


def preprocess(config, path, num_logs = None):
    # input: string of path to csv file
    # output: prints lines consisting of timestamp and query (comma separated)

    enum_dict = GetEnumDict(config['schema'])

    processed_queries = 0

    f = gzip.open(path, mode='rt')
    reader = csv.reader(f, delimiter=',')
    
    for query_info in reader:
        processed_queries += 1

        if (not num_logs is None) and processed_queries > num_logs:
            break

        if config['name'] == 'tiramisu':
            time_stamp = query_info[0]
            time_stamp = time_stamp[: -8] # remove milliseconds and the time zone

        else:
            if query_info[config['type_index']] != 'Query':  # skip if not a query
                continue

            # create timestamp
            if config['name'] == 'admission':
                day = query_info[0]
                time = query_info[1].split(".")[0]  # removes the milliseconds
                time_stamp = day + " " + time

            if config['name'] == 'oli':
                time_stamp = query_info[0]
                if time_stamp[7] == ' ':
                    time_stamp = time_stamp[0: 7] + '0' + time_stamp[8: -1]
        #IF

        time_stamp = datetime.datetime.strptime(
            time_stamp, config['time_stamp_format'])
        time_stamp = time_stamp.replace(second=0)  # accurate to the minute
        # Format query
        query = query_info[config['query_index']]

        for stmt in STATEMENTS:
            idx = query.find(stmt)
            if idx >= 0:
                break

        if idx < 0:
            continue

        # put back all the params for unnamed prepared statements...
        # this is nasty...
        if (not config['mysql']) and "execute" in query:
            params = query_info[config['param_index']]
            params = re.findall("'.+?'", params)
            for i, param in reversed(list(enumerate(params))):
                query = query.replace("${}".format(i + 1), param)

        query = query[idx:]
        if query[-1] != ";":
            query += ";"

        for clean_enum in enum_dict:
            query = query.replace(clean_enum, enum_dict[clean_enum])

        print(str(time_stamp) + ',' + query_info[config['conn_pos']] + ',' + query)

if __name__ == '__main__':
    preprocess(PROJECTS[sys.argv[1]], sys.argv[2], 1100000000)





