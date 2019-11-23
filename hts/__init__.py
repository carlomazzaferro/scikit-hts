# -*- coding: utf-8 -*-
import boto3
import logging

__author__ = """Carlo Mazzaferro"""
__email__ = 'carlo.mazzaferro@u.group'
__version__ = '0.1.0'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3_resource = boto3.resource('s3', region_name='us-east-1')
s3_client = boto3.client('s3', region_name='us-east-1')
