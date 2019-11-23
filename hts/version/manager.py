from boto3.exceptions import S3UploadFailedError
from distutils.version import StrictVersion

from hts.config import ACCOUNT_ID, TRAINING_JOB_NAME, HAS_MODEL_REGEX, RUN_EXISTS_REGEX
from hts import s3_client


class VersionManager:
    bucket_name = 'sagemaker-us-east-1-{}'.format(ACCOUNT_ID)
    version_bucket = TRAINING_JOB_NAME + '-version'

    @classmethod
    def largest_version(cls, ver_list):
        if not ver_list:
            return None
        return max(ver_list, key=lambda x: StrictVersion(cls.to_default(x)))

    @staticmethod
    def to_canonical(v):
        return v.replace('.', '-')

    @staticmethod
    def to_default(v):
        return v.replace('-', '.')

    @classmethod
    def get_latest_run(cls, valid=True):

        regex = HAS_MODEL_REGEX if valid else RUN_EXISTS_REGEX
        runs = []
        for item in cls.bucket_iterator():
            key = item['Key']
            if regex.match(key):
                ver = key.replace(TRAINING_JOB_NAME + '-', '').split('/')[0]
                runs.append(ver)
        return runs

    @classmethod
    def bucket_iterator(cls):
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=cls.bucket_name)

        for page in page_iterator:
            if page['KeyCount'] > 0:
                for item in page['Contents']:
                    yield item

    @staticmethod
    def bump(version):
        M, m, p = version.split('-')
        return '-'.join([M, str(int(m) + 1), p])

    @classmethod
    def publish(cls, version):
        with open('version', 'w') as ve:
            ve.write(version)
        s3_client.upload_file('version', TRAINING_JOB_NAME + '-version', 'version')

    @classmethod
    def get_version(cls, valid=False):
        last = cls.largest_version(cls.get_latest_run(valid=valid))
        if not last:
            with open('version', 'w') as v:
                v.write('1-0-0')
            try:
                s3_client.upload_file('version', cls.version_bucket, 'version')
            except S3UploadFailedError:
                s3_client.create_bucket(Bucket=cls.version_bucket)
                s3_client.upload_file('version', cls.version_bucket, 'version')
            return '1-0-0'

        else:
            return cls.bump(last)
