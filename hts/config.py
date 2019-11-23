import os
import re

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
DATA_DIR = os.path.join(ROOT_DIR, 'data')

ACCOUNT_ID = os.environ['AWS_ACCOUNT_ID']
TRAINING_JOB_NAME = os.environ['TRAINING_JOB_NAME']

SAGEMAKER_ROLE = 'SageMaker_circ_dev'
INSTANCE_TYPE = 'ml.c4.8xlarge'

REGION_MAPPING = {'dev': 'us-east-2', 'test': 'us-west-1', 'prod': 'us-west-2', 'local': 'us-east-2'}

HAS_MODEL_REGEX = re.compile(r'{}-([0-9]\d*)-([0-9]\d*)-([0-9]\d*)\/output\/model.tar.gz'.format(TRAINING_JOB_NAME))
RUN_EXISTS_REGEX = re.compile(r'{}-([0-9]\d*)-([0-9]\d*)-([0-9]\d*)\/source\/sourcedir.tar.gz'.format(TRAINING_JOB_NAME))
