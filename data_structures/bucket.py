class Bucket:

    def __init__(self, bucket_size, bucket_id):
        self.bucket_list = []
        self.bucket_size = bucket_size
        self.id = bucket_id

    def get_list(self):
        return self.bucket_list

    def get_id(self):
        return self.id


if __name__ == '__main__':
    print('test bucket')
    bucket = Bucket(5, 1)
    print('finish test')
