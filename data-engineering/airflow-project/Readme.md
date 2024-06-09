# Airflow

## What is Airflow

An open-source platform to create, schedule, monitor tasks and workflows. It's popular because it's open source, uses python, it has a lot of plugins for aws, azure, gcp, snowflake, databricks and on premises servers, it is highly scalable and has a rich interface for troubleshooting. It can be used with any batch data processing. It's a workflow management solution for orcastration and monitoring.

DAG is a workflow or a Directed, Acyclic, Graph; meaning dependencies have specified directions, there's no loops and diagram consits of nodes (tasks) and edges (dependencies).

DAGs are written in python and consit of one or more tasks, each task is defined by an Operator and Parameters,

```python

#daily_elt_run.py
from airflow import DAG
from airflow.providers.amazon.aws.transfers.sql_to_s3 import SQLToS3Operator
from airflow.providers.snowflake.transfers.s3_to_snowflake import S3ToSnowflakeOperator
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago

#Define the Dag schedule
dag = DAG(
    dag_id='daily_elt_run',
    default_args={'start_date': days_ago(1)},
    schedule_interval='0 23 * * *' # Run at 23:00 daily
)

# Define the tasks
extract = SqlToS3Operator(
    task_id='Extract',
    sql_conn_id='conn_id_name',
    query='SELECT * FROM my_table',
    s3_bucket='my_bucket',
    s3_key='my_key',
    repace=True,
    dag=dag,
    doc='Extract data from the MySQL database my_database to the AWS S3 bucket my_bucket'
)

load = S3ToSnowflakeOperator(
    task_id="Load",
    snowflake_conn_id="SNOWFLAKE_CONN_ID",
    s3_keys=["my_s3_file_path"],
    table="SNOWFLAKE_SAMPLE_TABLE",
    schema="SNOWFLAKE_SCHEMA",
    stage="SNOWFLAKE_STAGE",
    file_format="(type = 'CSV', field_delimiter = ';')",
    dag=dag,
    doc="Load data from the AWS S3 bucket my_buvket to the Snowflake staging table staging_table"
)

transform = BashOperator(
    task_id='Transform',
    bash_command='source dbt_venv/activate && dbt build',
    cwd='dbt_proj',
)

# defines the dependencies or order they should be executed in
extract >> load >> transform
```

## Installing Airflow on Windows

Since Airflow does not work with windows it needs to be run within a Docker container. This [blog post](https://sleek-data.blogspot.com/2023/09/how-to-install-airflow-on-windows.html) gives more informaiton on how to create an [Apache Airflow Docker Image](https://hub.docker.com/r/apache/airflow).

Create docker file with the following info:

```console
FROM apache/airflow:latest

USER root

RUN apt-get update && \

    apt-get -y install git && \

    apt-get clean

USER airflow
```

Ensuring Docker Engine is running on windows and vs code has docker plugin, right click on the docker file in vs code and select build image, wait for image to be built.

Create a docker-compose.yml file with the following info:

```console
version: "3"

services:
  airflow:
    image: airflow:latest
    volumes:
      - ./airflow:/opt/airflow
    ports:
      - "8080:8080"

    command: airflow standalone

```

Now right click on docker-compose.yml file and select compose up
