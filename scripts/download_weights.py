import os

from tqdm import tqdm

from .utils import gdrive

file_ids = [
    '1oYrwr8N-mqLXy2WJyb-R9uVSJ-z1D5r0',
    '1vpVBSc1w8BD65vqK24S5pAQXhmaw73kW',
    '1wTAuGzy9I7n_FsrSTh_gF-JGOQKRdaoC',
    '1xr-gNGxpcmodt-PIciBRK-x8N9NfI13y',
    '1wPSHuXwbOgZumiX1X3-xTJp1xR1t1xRg',
    '1C8Jxxwx_O7gNzkPrRzbdeWUkHIHAxV1Y',
    '1TXEQeN6s-ptymqxgupByL1UEizxXU-Lv',
    '1K-wRy4v68ThXPRKSXv3XcJ_Sb3KNraUU',
    '1wwef-z1hj6hafUgtbVMB6p6_UIaBOW7N',
]

for file_id in tqdm(file_ids):
    gdrive.download_file_from_google_drive(file_id, destination='pretrained_weights/azimuth')