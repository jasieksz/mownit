import csv

maxInt = 2129407327


class persistenceManager():
    def __init__(self):
        csv.field_size_limit(maxInt)

    def map2csv(self, path, fields, inMap):
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for k, v in inMap.items():
                writer.writerow({fields[0]: k, fields[1]: v})

    def csv2map(self, path, fields):
        newMap = {}
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                k = row[fields[0]]
                v = row[fields[1]]
                newMap[k] = v
        return newMap

    def csv2OneRow(self, path, fields):
        newMap = {}
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            i = 0
            for row in reader:
                k = row[fields[0]]
                v = row[fields[1]]
                newMap[k] = v
                i += 1
                if (i > 2):
                    break

        return newMap
