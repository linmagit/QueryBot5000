# QueryBot 5000
**QueryBot 5000 (QB5000)** is a  robust forecasting framework that allows a DBMS to predict the expected arrival rate of queries
in the future based on historical data. This is the source code for our 
[SIGMOD paper](http://www.cs.cmu.edu/~malin199/publications/2018.forecasting.sigmod.pdf): **_Query-based Workload Forecasting for Self-Driving Database Management Systems_**.

## Framework Pipeline:

### Anonymization
We first anonymize all the queries from the real-world traces used in our experiments for privacy purposes. The components below use the anonymization results from this step as their input.

    cd anonymizer
    ./log-anonymizer.py --help
    
### Pre-processor
This component extracts the **template**s from the anonymized queries and records the arrival rate history for each template.

    cd pre-processor
    ./templatizer.py --help
    
### Clusterer
This component groups query templates with similar arrival rate patterns into **cluster**s.

    cd clusterer
    ./online_clustering.py --help
_generate-cluster-coverage.py_ generates the time series for the largest _MAX_CLUSTER_NUM_ clusters on each day, which are used in the forecasting evaluation.
    
### Forecaster
This component uses a combination of linear regression, recurrent neural network, and kernel regression to predict the arrival rate pattern of each query cluster on different prediction **horizon**s and **interval**s.

    cd forecaster
    ./exp_multi_online_continuous.py --help
 
### Workload Simulator
This simulator populates a synthetic database with a given schema file, removes all the secondary indexes, replays the query trace of the workload, and builds appropriate indexes with the real-time workload forecasting results.

    cd workload-simulator
    ./workload-simulator.py --help
    
## NOTE
This repo does not have an end-to-end running framework. We build different components separately and pass the results through a workload simulator that connects to MySQL/PostgreSQL for experimental purposes. We are integrating the full framework into [Peloton](http://pelotondb.io/) Self-Driving DBMS. Please check out our [source code](https://github.com/cmu-db/peloton/tree/master/src/include/brain) there for more reference.

## License
    Copyright 2018, Carnegie Mellon University

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
