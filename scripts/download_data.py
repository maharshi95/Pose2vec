import os

from .utils import gdrive
data_file_id = '1I64s0OHllJRT-eBDp8fqAOhT5HBmf4oC'
destination_file = 'data/pose_data.mat'

if not os.path.exists(destination_file):
    gdrive.download_file_from_google_drive(data_file_id, destination_file)
else:
    print 'File already exists.. Do you want to resume with the download and replace it ? (Y/N)'
    user_response = raw_input('').strip().lower()
    valid_reponse = len(user_response) == 1 and user_response[0] in {'n', 'y'}
    while not valid_reponse:
        user_response = raw_input('Invalid reponse. Give Y/N as a valid reponse.').strip().lower()
        valid_reponse = len(user_response) == 1 and user_response[0] in {'n', 'y'}
    if user_response == 'y':
        print 'Downloading the file.'
        gdrive.download_file_from_google_drive(data_file_id, destination_file)
    else:
        print 'Skipping the download'