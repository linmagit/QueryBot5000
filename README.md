# QueryBot 5000
**QueryBot 5000 (QB5000)** is a  robust forecasting framework that allows a DBMS to predict the expected arrival rate of queries
in the future based on historical data. This is the source code for our 
[SIGMOD paper](http://www.cs.cmu.edu/~malin199/publications/2018.forecasting.sigmod.pdf): **_Query-based Workload Forecasting for Self-Driving Database Management Systems_**.

## Run forecasting on a sample of BusTracker workload:
    ./run.sh
We provide an example of the workload forecasting for a sample subset of **BusTracker** workload. The prediction specified in the script is on 1 hour interval and 3 day horizon. The predicted arrival rates with different models for each cluster are in the _prediction-results_ folder. All the query templates of the workload can be found at _templates.txt_.

The default experimental setting is to run under CPU. If you have a GPU, you can change [this parameter](https://github.com/malin1993ml/QueryBot5000/blob/master/forecaster/exp_multi_online_continuous.py#L101) to _True_ to enable GPU training.

### Dependencies
python>=3.5
scikit-learn>=0.18.1
sortedcontainers>=1.5.7
statsmodels>=0.8.0
scipy>=0.19.0
numpy>=1.14.2
matplotlib>=2.0.2
pytorch>=0.2.0_1 (you need to install the GPU version if you want to use GPU)

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
    
## Inquiry about Data
Due to legal and privacy constraints, unfortunately we cannot publish the full datasets that we used in the experiments for the publication (especially for the two student-related **Admissions** and **MOOC** workloads). To the best of our effort, we managed to publish a subset (2% random sampling) of the **BusTracker** workload trace and the schema file here:
http://www.cs.cmu.edu/~malin199/data/tiramisu-sample/

We use [this script](https://github.com/malin1993ml/QueryBot5000/blob/master/anonymizer/run-sampler.sh) to generate the sample subset of the original workload trace.
    
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
