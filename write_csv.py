import glob
import csv

file_name = []

name = 'submission.csv'
with open(name, 'w') as f:
    writer = csv.writer(f, delimiter=',', lineterminator='\n')
    writer.writerow(['guid/image', 'label'])
    for file in glob.glob('test-20211123T043833Z-*/test/*/*_image.jpg'):
        row_info = file.split('\\')[2] + '/' + file.split('\\')[3][0:4]
        writer.writerow([row_info])
