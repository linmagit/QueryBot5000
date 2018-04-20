#!/usr/bin/env python
# coding: utf-8


import collections
import numpy as np
np.set_printoptions(threshold=np.inf)
import re

KEYWORDS = ["KEY", "PRIMARY", "UNIQUE", "CONSTRAINT"]

def extract_tables_and_columns(sql_dump):
    """ Extracts the tables and columns from the sql_dump and
    puts them into the semantics dictionary passed in

    Args:
       semantics_dict (dict): dictionary to populate
       sql_dump (iterable): iterable containing the sql_dump

    Returns:
        altered semantics dictionary

    """
    semantics_dict = collections.OrderedDict()
    table = None
    for line in sql_dump:
        table_name = re.match("CREATE TABLE (\`*.+\`*)\(.*", line)
        if table_name is not None:
            table = (table_name.group(1)).replace("(", "").replace("`", "").strip()
            semantics_dict[table] = collections.OrderedDict()
            semantics_dict[table]["num_of_accesses"] = 0
            continue
        elif table is not None:
            line = line.strip()
            # Note: this is table not table_name, like above
            column = re.match("\`*(\w+)\`*", line)
            if column is not None:
                key_2 = column.group(0).replace("`", "").strip()

                # Handle case of PRIMARY KEY, KEY, and UNIQUE KEY
                is_key = False
                for key in KEYWORDS:
                    if key in key_2:  #
                        is_key = True
                if is_key:
                    continue

                (semantics_dict[table])[key_2] = 0
            else:  # inside a table but nothing is declared? Exit
                end_paren = re.match("\)", line)
                if end_paren is not None:
                    table = None
        else:
            continue
    print("Populated schema dict...")
    return semantics_dict
